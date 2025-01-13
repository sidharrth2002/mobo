

import logging
import pickle
import time
from typing import Dict, Optional
import numpy as np
from ray.tune.schedulers.trial_scheduler import FIFOScheduler, TrialScheduler
from ray.util import PublicAPI
from ray.tune.experiment import Trial

logger = logging.getLogger(__name__)
    
@PublicAPI
class MultiObjectiveAsyncHyperBandScheduler(FIFOScheduler):
    """
        time_attr: A training result attr to use for comparing time.
            Note that you can pass in something non-temporal such as
            `training_iteration` as a measure of progress, the only requirement
            is that the attribute should increase monotonically.
        max_t: max time units per trial. Trials will be stopped after
            max_t time units (determined by time_attr) have passed.
        grace_period: Only stop trials at least this old in time.
            The units are the same as the attribute named by `time_attr`.
        reduction_factor: Used to set halving rate and amount. This
            is simply a unit-less scalar.
        brackets: Number of brackets. Each bracket has a different
            halving rate, specified by the reduction factor.
        stop_last_trials: Whether to terminate the trials after
            reaching max_t. Defaults to True.
    """
    def __init__(
        self,
        objectives,
        time_attr="training_iteration",
        max_t=100,
        grace_period=1,
        reduction_factor=4,
        brackets=1,
        # strategy=MOSelectionStrategies.EPS_NET,
        stop_last_trials=True
    ):
        # assert strategy in [MOSelectionStrategies.EPS_NET, MOSelectionStrategies.NSGA_II], "Invalid selection strategy"
        assert all(
            [o in ["min", "max"] for o in objectives.values()]
        ), "Objectives must be 'min' or 'max'."
        
        self.objectives = objectives
        self.time_attr = time_attr
        self.max_t = max_t
        self.grace_period = grace_period
        self.reduction_factor = reduction_factor
        # self.strategy = strategy
                        
        # store the trials
        self._trial_info = {}
        
        # Tracks state for new trial add
        self._brackets = [
            _Bracket(
                min_t=grace_period,
                max_t=max_t,
                reduction_factor=reduction_factor,
                s=s,
                stop_last_trials=stop_last_trials,
            )
            for s in range(brackets)
        ]
        
        self._counter = 0
        self._num_stopped = 0
        self._time_attr = time_attr
        self._stop_last_trials = stop_last_trials
        
        # standardise objectives to be max-oriented
        self._metric_op = self.__prepare_sign_vector(objectives)
        
    def on_trial_add(self, tune_controller: "TuneController", trial: Trial):
        sizes = np.array([len(b._rungs) for b in self._brackets])
        probs = np.e ** (sizes - sizes.max())
        normalized = probs / probs.sum()
        idx = np.random.choice(len(self._brackets), p=normalized)
        self._trial_info[trial.trial_id] = self._brackets[idx]

    def __prepare_sign_vector(self, objectives):
        """
        Prepares a sign vector to handle user-defined "min" or "max" objectives.

        For each objective 'obj':
        - If user says "max", we return +1  (since 'bigger is better')
        - If user says "min", we return -1  (to flip it so that 'bigger is better' internally)

        Example:
        objectives = {"accuracy": "max", "latency": "min"}
        => sign_vector = [ +1, -1 ]
        
        After we multiply the raw [accuracy, latency] by [1, -1],
        we end up with [accuracy, -latency], so now both are effectively
        "larger is better" in the transformed space.
        """
        converter = {"min": -1.0, "max": 1.0}
        try:
            # For each objective key, pick +1 if "max", -1 if "min"
            sign_vector = np.array([converter[objectives[k]] for k in objectives])
        except KeyError:
            raise ValueError("Objectives must be either 'min' or 'max'.")
        return sign_vector

    def on_trial_result(
        self, tune_controller: "TuneController", trial: Trial, result: Dict
    ) -> str:
        action = TrialScheduler.CONTINUE
        
        # make sure we have all the objectives in order to the processing
        if self._time_attr not in result or not all(obj in result for obj in self.objectives):
            return action
        
        if result[self._time_attr] >= self.max_t and self._stop_last_trials:
            action = TrialScheduler.STOP
            print("Trial stopped due to max_t")
        else:
            bracket = self._trial_info[trial.trial_id]
            
            metrics = np.array([result[obj] for obj in self.objectives]) * self._metric_op
            
            action = bracket.on_result(
                trial, result[self._time_attr], metrics
            )
            print("Trial {}, iter {} metrics {} action: {}".format(trial.trial_id, result[self._time_attr], metrics, action))

        if action == TrialScheduler.STOP:
            self._num_stopped += 1
        
        return action
    
    def on_trial_complete(
        self, tune_controller: "TuneController", trial: Trial, result: Dict
    ):
        if self._time_attr not in result or not all(obj in result for obj in self.objectives):
            return
        bracket = self._trial_info[trial.trial_id]
        
        metrics = np.array([result[obj] for obj in self.objectives]) * self._metric_op
        
        bracket.on_result(
            trial, result[self._time_attr], metrics
        )
        del self._trial_info[trial.trial_id]
        
    def on_trial_remove(self, tune_controller: "TuneController", trial: Trial):
        del self._trial_info[trial.trial_id]
    
    def debug_string(self) -> str:
        out = "Using MO-ASHA: num_stopped={}".format(self._num_stopped)
        out += "\n" + "\n".join([b.debug_str() for b in self._brackets])
        return out
    
    def save(self, checkpoint_path: str):
        save_object = self.__dict__
        with open(checkpoint_path, "wb") as outputFile:
            pickle.dump(save_object, outputFile)

    def restore(self, checkpoint_path: str):
        with open(checkpoint_path, "rb") as inputFile:
            save_object = pickle.load(inputFile)
        self.__dict__.update(save_object)

class _Bracket:
    """
    MultiObjectiveBracket
    
    Rungs are created in reversed order so that we can more easily find
    the correct rung corresponding to the current iteration of the result.
    """
    def __init__(
        self,
        min_t: int,
        max_t: int,
        reduction_factor: float,
        s: int,
        stop_last_trials: bool = True
    ):
        self.rf = reduction_factor
        MAX_RUNGS = int(np.long(max_t / min_t) / np.log(self.rf) - s + 1)
        self._rungs = [
            (min_t * self.rf ** (k + s), {}) for k in reversed(range(MAX_RUNGS))
        ]
        self._stop_last_trials = stop_last_trials
        # to compute pareto, need all past points, so store it
        self.all_points = []

    def dominates(self, point_a, point_b):
        # Example for 2D objectives: must be at least as good in all
        # and strictly better in at least one
        # Adjust to your dimensionality and direction (maximize/minimize)
        a_obj = point_a[1]
        b_obj = point_b[1]
        better_or_equal = all(a >= b for a, b in zip(a_obj, b_obj))
        strictly_better = any(a > b for a, b in zip(a_obj, b_obj))
        return better_or_equal and strictly_better

    # def compute_pareto_front(self):
    #     # naive implementation        
    #     non_dominated = []
    #     print("All points: {}".format(self.all_points))
    #     for i, pt_i in self.all_points.items():
    #         dominated = False
    #         for j, pt_j in self.all_points.items():
    #             if i != j and self.dominates(pt_j, pt_i):
    #                 dominated = True
    #                 break
    #         if not dominated:
    #             non_dominated.append(pt_i)
    #     return non_dominated
    
    def compute_pareto_front(self):
        # naive implementation
        non_dominated = []
        for i, pt_i in enumerate(self.all_points):
            dominated = False
            for j, pt_j in enumerate(self.all_points):
                if i != j and self.dominates(pt_j, pt_i):
                    dominated = True
                    break
            if not dominated:
                non_dominated.append(pt_i)
        return non_dominated
    
    def is_x_in_y(self, x, y):
        # x is a tuple, y is a list of tuples
        for y_i in y:
            print(x[0])
            equality_0 = x[0] == y_i[0]
            equality_1 = (x[1] == y_i[1]).all()
            if equality_0 and equality_1:
                return True

        return False

    def on_result(self, trial: Trial, cur_iter: int, cur_rew: np.ndarray) -> str:
        action = TrialScheduler.CONTINUE
        
        self.all_points.append((trial.trial_id, cur_rew))
        # only keep the latest result for each trial        
        # upon more thinking, looks like a later epoch will always dominate an earlier one
        # given the 
        # self.all_points[trial.trial_id] = cur_rew
        
        for milestone, recorded in self._rungs:
            if (
                cur_iter >= milestone
                and trial.trial_id in recorded
                and not self._stop_last_trials
            ):
                # if result recorded for trial already, decision to continue training already made
                # skip new pareto calculate and just continue training
                break
            if cur_iter < milestone or trial.trial_id in recorded:
                continue
            else:
                # make promotion decision based on pareto front
                pareto_front = self.compute_pareto_front()
                print("Pareto front: {}".format(pareto_front))
                if not self.is_x_in_y((trial.trial_id, cur_rew), pareto_front):
                    action = TrialScheduler.STOP
                if cur_rew is None:
                    logger.warning(
                        "Reward attribute is None"
                    )
                else:
                    recorded[trial.trial_id] = cur_rew
    
        return action
    
    def debug_str(self) -> str:
        iters = " | ".join(
            [
                "Iter {:.3f}: {}".format(milestone, self.compute_pareto_front())
                for milestone, recorded in self._rungs
            ]
        )
        return "Bracket: " + iters
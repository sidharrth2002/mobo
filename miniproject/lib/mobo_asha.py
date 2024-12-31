"""
Sidharrth Nagappan
A Rudimentary Multi-Objective Asynchronous Successive Halving Scheduler (MO-ASHA)

Loosely based on Autogluon Implementation of MO-ASHA and Schmucker et al. (2021)'s paper on MO-ASHA
"""
import logging
import time
import numpy as np
from ray.tune.schedulers.trial_scheduler import FIFOScheduler, TrialScheduler
from ray.util import PublicAPI

logger = logging.getLogger(__name__)

class MOSelectionStrategies:
    # farthest first coverage among best solutions
    EPS_NET = "eps_net"
    # classic multi-objective ranking via non-dominated sorting + crowding distance
    NSGA_II = "nsga_ii"

@PublicAPI
class MultiObjectiveAsyncHyperBandScheduler(FIFOScheduler):
    """Multi-Objective Async Successive Halving Scheduler.

    Implements a multi-objective ASHA scheduler using strategies like EPS-NET and NSGA-II for trial promotion based on Pareto fronts.

    Args:
        objectives: Dictionary with objective names as keys and "max" or "min"
            as values to specify whether to maximize or minimize each objective.
        time_attr: Attribute used for resource allocation, e.g., "training_iteration".
        max_t: Maximum resource level.
        grace_period: Minimum resource allocation before stopping.
        reduction_factor: Resource reduction factor for successive halving.
        strategy: Promotion strategy, either "eps_net" or "nsga_ii".
    """

    def __init__(
        self,
        objectives,
        time_attr="training_iteration",
        max_t=100,
        grace_period=1,
        reduction_factor=4,
        strategy=MOSelectionStrategies.EPS_NET,
    ):
        assert strategy in [MOSelectionStrategies.EPS_NET, MOSelectionStrategies.NSGA_II], "Invalid selection strategy"
        assert all(
            [o in ["min", "max"] for o in objectives.values()]
        ), "Objectives must be 'min' or 'max'."

        super().__init__()
        
        self.objectives = objectives
        self.time_attr = time_attr
        self.max_t = max_t
        self.grace_period = grace_period
        self.reduction_factor = reduction_factor
        self.strategy = strategy
        
        # standardise objectives to be max-oriented
        self.sign_vector = self.__prepare_sign_vector(objectives)
        
        # store brackets
        # raytune implementation uses default single bracket (after they discussed with the authors of ASHA)
        self.brackets = []
        
        # store the trials
        self.trial_info = {}
        
        # initialise the rung system
        self._init_brackets()
        
    def _init_brackets(self):
        """
        Compute rung levels from grace_period, max_t, reduction_factor.
        rung_levels = [grace_period * reduction_factor ** i for i in range(num_rungs)]
        """
        num_rungs = int(np.log(self.max_t / self.grace_period) / np.log(self.reduction_factor)) + 1
        rung_levels = [
            self.grace_period * (self.reduction_factor ** i) for i in range(num_rungs)
        ]

        # Create a single bracket containing rung info
        self.brackets = [
            MultiObjectiveBracket(rung_levels, self.strategy, self.sign_vector)
        ]
        
    def on_trial_add(self, tune_controller, trial):
        """
        Add a trial to the bracket. 
        Since there's only one bracket in this implementation, nothing much to do here.
        """
        self.trial_info[trial.trial_id] = self.brackets[0]
    
    def on_trial_result(self, tune_controller, trial, result):
        """
        Called when trial reports intermediate results.
        In my case, it's when the PyTorch Lightning callback logs metrics to Tune at the end of each epoch (aka iteration)
        
        1. Check if all objectives present
        2. Is the resource >= rung level?
        3. Is the trial dominated?
        
        If dominated, STOP early, if not let continue, and it will naturally get promoted to next rung later.
        """
        action = TrialScheduler.CONTINUE
        bracket = self.trial_info[trial.trial_id]
        
        # all objectives required to make a decision
        if self.time_attr not in result or not all(obj in result for obj in self.objectives):
            return action
    
        resource = result[self.time_attr]
        
        # multiply by sign vector to standardise to max-oriented
        metrics = np.array([result[obj] for obj in self.objectives]) * self.sign_vector
        
        # bracket decides whether to STOP or CONTINUE
        action = bracket.on_result(trial, resource, metrics)
        return action
    
    def on_trial_complete(self, tune_controller, trial, result):
        """
        Called when trial finishes (either by STOP or naturally by completing all epochs).
        Do a final rung check to make sure it's recordered, then remove from trial_info.
        """
        bracket = self.trial_info.get(trial.trial_id)
        if bracket:
            resource = result[self.time_attr]
            # Convert final metrics
            metrics = np.array([result[obj] for obj in self.objectives]) * self.sign_vector
            bracket.on_result(trial, resource, metrics)
            del self.trial_info[trial.trial_id]
    
    def on_trial_remove(self, tune_controller, trial):
        """
        Called when trial removed (cancelled outside of Tune).
        """
        if trial.trial_id in self.trial_info:
            del self.trial_info[trial.trial_id]
    
    def debug_string(self):
        """
        Returns a string to describe the current state of the scheduler.
        """
        return "\n".join([bracket.debug_string() for bracket in self.brackets])
    
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
    
class MultiObjectiveBracket:
    """
    A bracket for multi-objective ASHA. Each bracket has rung_levels (e.g. [1, 3, 9, ...])
    
    For each rung, dict is stored {trial_id: metrics}
    
    When trial hits rung i:
    1. Gather all rung i's existing solutions
    2. Filter out dominated solutions
    3. Reorder them by EPS_NET (default) or NSGA-II
    4. Check if new trial is dominated. If dominated, stop. If not, continue.
    """
    def __init__(self, rung_levels, strategy, sign_vector):
        # rung levels looks something like [1, 3, 9, 27]
        self.rung_levels = rung_levels
        self.strategy = strategy
        self.sign_vector = sign_vector
        
        self.rungs = [
            {"level": level, "recorded": {}} for level in rung_levels
        ]
        print(f"Rungs: {self.rungs}")
        
    def on_result(self, trial, resource, metrics):
        """
        Called when trial reports result with 'resource' >= rung level.
        Record trial's metrics in the rung and check if it's dominated.
        If dominated, stop, else, let it continue.
        Doesn't completely stay true to form compared to the original MO-ASHA implementation.
        """
        action = TrialScheduler.CONTINUE
        
        time_start = time.time()
        # iterate over rung levels
        for rung in self.rungs:
            
            # if not yet reached rung level, continue
            if resource < rung["level"]:
                continue
        
            recorded = rung["recorded"]
            
            # if already recorded trial, skip, so there's no repeated record
            if trial.trial_id in recorded:
                continue
            
            # filter existing rung solutions, remove dominated and reorder
            candidates = self._calculate_candidates(recorded)
            
            # check if new trial dominated by rung's best solutions
            if candidates is not None and not self._is_promotable(metrics, candidates):
                print(f"[Rung={rung['level']}] Trial {trial.trial_id} is dominated. -> STOP")
                action = TrialScheduler.STOP
            
            # record the trial in the rung
            recorded[trial.trial_id] = metrics
            
            # only record in first rung for which resources >= rung["level"], then go and break
            break
        
        time_end = time.time()
        
        # I was initially scared that this was bottlenecking the process
        # Turns out to be pretty fast lol
        print(f"Time taken for on_result: {time_end - time_start}")
        
        return action
    
    def _calculate_candidates(self, recorded):
        """
        From rung["recorded"], produce set of best solutions:
        1. Convert recorded.values() => NxD array
        
        (shape) -> (#trials_in_rung, #objectives)
        
        2. Filter out dominated solutions
        3. Reorder by EPS_NET or NSGA-II
        4. Return the NxD array in new order
        """
        if not recorded:
            return None
        
        # points is 2D array of shape
        points = np.array(list(recorded.values()))
        
        # Identify the subset of non-dominated points
        # These are the points that we should compare new trials against
        nd_indices = self._non_dominated_indices(points)
        nd_points = points[nd_indices]
        
        # reorder via either EpsNet or NSGA-II
        if self.strategy == MOSelectionStrategies.EPS_NET:
            return self._eps_net(nd_points)
        elif self.strategy == MOSelectionStrategies.NSGA_II:
            return self._nsga_ii(nd_points)
    
    def _is_promotable(self, new_metrics, candidates):
        """
        Returns True if new_metrics not dominated by any row in candidates.
        Row c in candidates dominates new metrics if c >= new_metrics in all dims and c > new_metrics in at least 1 dim
        (bigger is better)
        """
        for c in candidates:
            if np.all(c >= new_metrics) and np.any(c > new_metrics):
                # new metrics has been dominated, not promotable
                return False
        return True
    
    def _non_dominated_indices(self, points):
        """
        Return indices of non-pdominated points in 'points'. After sign flipping, bigger is better in each dimension. So p[i] dominates p[j] if p[i] >= p[j] in every dimension and p[i] > p[j] in at least one dimension.
        
        This is standard (very rudimentary) Pareto filter, eliminate all p[k] strictly worse than some p[i].
        """
        n = len(points)
        # True means keep, False means dominated
        mask = np.ones(n, dtype=bool)
        for i in range(n):
            # if p[i] already found dominated, then skip
            if not mask[i]:
                continue
            for j in range(n):
                if i == k or not mask[j]:
                    continue
                
                if (np.all(points[i] >= points[j]) and np.any(points[i] > points[j])):
                    mask[j] = False
                
        return np.where(mask)[0]
    
    def _eps_net(self, points):
        """
        Farthest-first approach
        1. Start with index 0
        2. Iteratively pick point that has largest min-distance to the already selected set, to ensure wide coverage
        """
        if len(points) == 0:
            return None
        
        selected = [0]
        remaining = list(range(1, len(points)))
        
        while remaining:
            # for each candidate i in remaining, compute distance to nearest selected
            distances = []
            for i in remaining:
                # dist from points[i] to closest among selected
                min_dist = np.min([np.linalg.norm(points[i] - points[j]) for j in selected])
                distances.append(min_dist)
            
            # farthest candidate from any selected is appended
            idx = np.argmax(distances)
            chosen_index = remaining.pop(idx)
            selected.append(chosen_index)
            
        # return them in farthest-first order
        return points[selected]
    
    def _nsga_ii(self, points):
        """
        Non-dominated sorting => front layers => within each front, sort by the crowding distance.
        Return all points in order of front+crowding.
        """
        if len(points) == 0:
            return None
        
        # use fast dominated sort to 
        fronts = self._fast_dominated_sort(points)
        result_indices = []
        for front in fronts:
            # crowding distance for that front
            cdist = self._crowding_distance(points[front])
            # sort front in descending order of cdist => prefer more spread out
            sorted_front = sorted(front, key=lambda x: -cdist[x])
            result_indices.extend(sorted_front)
        
        return points[result_indices]

    def _fast_nondominated_sort(self, points):
        """
        Identify front layers:
        - front 1 => no points dominate it
        - front 2 => dominated by front 1 only
        """
        n = len(points)
        # how many dominate me
        domination_counts = np.zeros(n, dtype=int)
        # who do I dominate
        dominated = [[] for _ in range(n)]
        
        # build the domination relationships
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                # i dominated by j if j >= i in all dims and j > i in at least one dim
                if (np.all(points[j] >= points[i]) and np.any(points[j] > points[i])):
                    domination_counts[i] += 1
                    dominated[j].append(i)
        
        fronts = [[]]
        for i in range(n):
            if domination_counts[i] == 0:
                fronts[0].append(i)
                
        # BFS expansion, pick front k, reduce dom_counts of those they dominate
        i = 0
        while len(fronts[i]) > 0:
            next_front = []
            for idx in fronts[i]:
                for d_idx in dominated[idx]:
                    domination_counts[d_idx] -= 1
                    if domination_counts[d_idx] == 0:
                        next_front.append(d_idx)
            i += 1
        
        return fronts[:-1]

    def _crowding_distance(self, points):
        """
        For each front, compute crowding distance that measures how isolated points is.
        Front then sorted by descending distance => more spread out solutions come first.
        """
        n = len(points)
        if n == 0:
            return []
        
        dists = np.zeros(n)
        num_dims = points.shape[1]
        
        for dim in range(num_dims):
            # sort by ascending values in dim
            sorted_idx = np.argsort(points[:, dim])
            # boundary points get infinite crowding distance
            dists[sorted_idx[0]] = np.inf
            dists[sorted_idx[-1]] = np.inf
            
            min_val = points[sorted_idx[0], dim]
            max_val = points[sorted_idx[-1], dim]
            diff = max_val - min_val
            
            if diff == 0:
                # all points same in this dimension, crowding distance doesn't change
                continue
            
            # for interior, measure how large gap is
            for rank_idx in range(1, n-1):
                lower = points[sorted_idx[rank_idx - 1], dim]
                upper = points[sorted_idx[rank_idx + 1], dim]
                dists[sorted_idx[rank_idx]] += (upper - lower) / diff
            
        return dists
    
    def debug_string(self):
        """
        Summarize how many trials are stored in each rung.
        Example: "Level 1.0: 4 trials | Level 3.0: 2 trials | ..."
        """
        rung_info = [
            f"Level {r['level']}: {len(r['recorded'])} trials"
            for r in self.rungs
        ]
        return " | ".join(rung_info)        
import logging
import numpy as np
import copy
from ray.tune.experiment import Trial
from ray.tune.schedulers.trial_scheduler import FIFOScheduler, TrialScheduler
from ray.util import PublicAPI

logger = logging.getLogger(__name__)

EPS_NET = "eps_net"
NSGA_II = "nsga_ii"

def prepare_sign_vector(objectives):
    """Generates a numpy vector to flip signs of objectives for minimization.

    Args:
        objectives: Dictionary specifying "min" or "max" for each objective.

    Returns:
        sign_vector: Numpy array where 1 corresponds to maximization, -1 to minimization.
    """
    converter = {
        "min": -1.0,
        "max": 1.0
    }
    try:
        sign_vector = np.array([converter[objectives[k]] for k in objectives])
    except KeyError:
        raise ValueError("Objectives must be 'min' or 'max'.")
    return sign_vector


@PublicAPI
class MultiObjectiveAsyncHyperBandScheduler(FIFOScheduler):
    """Multi-Objective Async Successive Halving Scheduler.

    Implements a multi-objective ASHA scheduler using strategies like EPS-NET
    and NSGA-II for trial promotion based on Pareto fronts.

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
        strategy=EPS_NET,
    ):
        assert strategy in [EPS_NET, NSGA_II], "Invalid selection strategy"
        assert all(
            v in ["max", "min"] for v in objectives.values()
        ), "Objectives must be 'max' or 'min'"

        super().__init__()

        self.objectives = objectives
        self.time_attr = time_attr
        self.max_t = max_t
        self.grace_period = grace_period
        self.reduction_factor = reduction_factor
        self.strategy = strategy
        # self.sign_vector = np.array(
        #     [-1 if v == "max" else 1 for v in objectives.values()]
        # )
        # TODO: JUST CHANGED, UNTESTED
        # added negative sign there
        self.sign_vector = -prepare_sign_vector(objectives)

        self.brackets = []
        self.trial_info = {}

        self._init_brackets()

    def _init_brackets(self):
        num_rungs = (
            int(np.log(self.max_t / self.grace_period) / np.log(self.reduction_factor))
            + 1
        )
        rung_levels = [
            self.grace_period * (self.reduction_factor**i) for i in range(num_rungs)
        ]

        self.brackets = [
            MultiObjectiveBracket(rung_levels, self.strategy, self.sign_vector)
            for _ in range(1)  # Single bracket in this example
        ]

    def on_trial_add(self, tune_controller, trial):
        self.trial_info[trial.trial_id] = self.brackets[0]

    def on_trial_result(self, tune_controller, trial, result):
        action = TrialScheduler.CONTINUE
        bracket = self.trial_info[trial.trial_id]
        if self.time_attr not in result or not all(
            obj in result for obj in self.objectives
        ):
            return action

        resource = result[self.time_attr]
        metrics = np.array([result[obj] for obj in self.objectives]) * self.sign_vector
        action = bracket.on_result(trial, resource, metrics)
        return action

    def on_trial_complete(self, tune_controller, trial, result):
        bracket = self.trial_info.get(trial.trial_id)
        if bracket:
            resource = result[self.time_attr]
            metrics = (
                np.array([result[obj] for obj in self.objectives]) * self.sign_vector
            )
            bracket.on_result(trial, resource, metrics)
            del self.trial_info[trial.trial_id]

    def on_trial_remove(self, tune_controller, trial):
        if trial.trial_id in self.trial_info:
            del self.trial_info[trial.trial_id]

    def debug_string(self):
        return "\n".join([bracket.debug_string() for bracket in self.brackets])


class MultiObjectiveBracket:
    def __init__(self, rung_levels, strategy, sign_vector):
        self.rung_levels = rung_levels
        self.strategy = strategy
        self.sign_vector = sign_vector
        self.rungs = [{"level": level, "recorded": {}} for level in rung_levels]

    def on_result(self, trial, resource, metrics):
        action = TrialScheduler.CONTINUE
        for rung in self.rungs:
            if resource < rung["level"]:
                continue

            recorded = rung["recorded"]
            if trial.trial_id in recorded:
                continue

            pareto_front = self._calculate_pareto_front(recorded)
            if pareto_front is not None and not self._is_promotable(
                metrics, pareto_front
            ):
                print(f"Trial {trial.trial_id} is not promotable with metrics {metrics} at resource {resource}")
                action = TrialScheduler.STOP

            recorded[trial.trial_id] = metrics
            break

        return action

    def _calculate_pareto_front(self, recorded):
        if not recorded:
            return None

        points = np.array(list(recorded.values()))
        if self.strategy == EPS_NET:
            return self._eps_net(points)
        elif self.strategy == NSGA_II:
            return self._nsga_ii(points)

    def _is_promotable(self, metrics, pareto_front):
        return not np.any(np.all(metrics >= pareto_front, axis=1))

    def _eps_net(self, points):
        selected = [0]
        # doesn't need to be a set
        remaining = list(range(1, len(points)))
        while remaining:
            distances = [
                np.min([np.linalg.norm(points[i] - points[j]) for j in selected])
                for i in remaining
            ]
            selected.append(remaining.pop(np.argmax(distances)))
        return points[selected]

    def _nsga_ii(self, points):
        fronts = self._fast_nondominated_sort(points)
        selected = []
        for front in fronts:
            crowding = self._crowding_distance(points[front])
            sorted_front = sorted(front, key=lambda x: -crowding[x])
            selected.extend(sorted_front)
        return points[selected]

    def _fast_nondominated_sort(self, points):
        n_points = points.shape[0]
        domination_counts = np.zeros(n_points, dtype=int)
        dominated = [[] for _ in range(n_points)]
        fronts = [[]]

        for i in range(n_points):
            for j in range(n_points):
                if np.all(points[i] <= points[j]) and np.any(points[i] < points[j]):
                    dominated[i].append(j)
                elif np.all(points[j] <= points[i]) and np.any(points[j] < points[i]):
                    domination_counts[i] += 1

            if domination_counts[i] == 0:
                fronts[0].append(i)

        i = 0
        while len(fronts[i]) > 0:
            next_front = []
            for j in fronts[i]:
                for k in dominated[j]:
                    domination_counts[k] -= 1
                    if domination_counts[k] == 0:
                        next_front.append(k)
            i += 1
            fronts.append(next_front)

        return fronts[:-1]

    def _crowding_distance(self, points):
        n_points = points.shape[0]
        distances = np.zeros(n_points)
        for i in range(points.shape[1]):
            sorted_indices = np.argsort(points[:, i])
            distances[sorted_indices[0]] = np.inf
            distances[sorted_indices[-1]] = np.inf
            for j in range(1, n_points - 1):
                distances[sorted_indices[j]] += (
                    points[sorted_indices[j + 1], i] - points[sorted_indices[j - 1], i]
                )
        return distances

    def debug_string(self):
        return " | ".join(
            [
                f"Level {rung['level']}: {len(rung['recorded'])} trials"
                for rung in self.rungs
            ]
        )

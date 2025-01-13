import logging
import time
import numpy as np
import copy
from ray.tune.experiment import Trial
from ray.tune.schedulers.trial_scheduler import FIFOScheduler, TrialScheduler
from ray.util import PublicAPI

logger = logging.getLogger(__name__)

# Two strategies for multi-objective selection:
EPS_NET = "eps_net"   # Farthest-first coverage among the best solutions
NSGA_II = "nsga_ii"   # Classic multi-objective ranking via non-dominated sorting + crowding distance

def prepare_sign_vector(objectives):
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


@PublicAPI
class MultiObjectiveAsyncHyperBandScheduler(FIFOScheduler):
    """
    Multi-Objective ASHA Scheduler.

    Extends Ray Tune's single-objective ASHA to multiple objectives by:
      - Transforming user objectives so that "bigger is better" for all
      - For each rung, deciding whether to STOP or CONTINUE a trial
        if it’s dominated by existing top solutions
      - Using either EpsNet or NSGA-II to reorder the set of best solutions

    Key points:
    1. time_attr: The resource dimension (e.g., "training_iteration" or epochs).
    2. objectives: Dictionary { objective_name: "max" or "min" }.
       We unify them so that after sign flipping, "bigger is better."
    3. strategy: Either "eps_net" or "nsga_ii".

    Example usage:
      scheduler = MultiObjectiveAsyncHyperBandScheduler(
          objectives={"acc": "max", "time": "min"},
          time_attr="training_iteration",
          max_t=50,
          grace_period=1,
          reduction_factor=3,
          strategy=EPS_NET
      )

    Then pass `scheduler` to Ray Tune's `tune.run(..., scheduler=scheduler)`.
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
        # Validate user inputs
        assert strategy in [EPS_NET, NSGA_II], "strategy must be eps_net or nsga_ii"
        assert all(v in ["max", "min"] for v in objectives.values()), \
            "Each objective must be either 'max' or 'min'"

        super().__init__()

        # Store config
        self.objectives = objectives              # e.g., {"acc": "max", "latency": "min"}
        self.time_attr = time_attr                # e.g., "training_iteration"
        self.max_t = max_t                        # e.g., 50
        self.grace_period = grace_period          # e.g., 1
        self.reduction_factor = reduction_factor  # e.g., 3 or 4
        self.strategy = strategy                  # e.g., "eps_net" or "nsga_ii"

        # sign_vector transforms raw objectives so that bigger=better
        # e.g., if "acc"="max", sign=+1; if "latency"="min", sign=-1
        self.sign_vector = prepare_sign_vector(objectives)

        # We store multiple "brackets" if needed; here we show a single bracket for simplicity
        self.brackets = []
        # Map each trial_id -> bracket
        self.trial_info = {}

        # Initialize rung system
        self._init_brackets()

    def _init_brackets(self):
        """
        Compute rung levels from grace_period, max_t, reduction_factor.
        Typically rung_levels = [grace_period * reduction_factor^0, ..., up to max_t].
        We create a single bracket of MultiObjectiveBracket in this example.
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
        Called when a new trial is added. We place the new trial in bracket[0].
        In a more sophisticated approach, we might do a random bracket assignment,
        but here we just store everything in the single bracket.
        """
        self.trial_info[trial.trial_id] = self.brackets[0]

    def on_trial_result(self, tune_controller, trial, result):
        """
        Called when a trial reports intermediate results, e.g. after each epoch.
        We check:
          - do we have the required time_attr and all objectives?
          - is the resource >= rung level?
          - is the trial dominated?

        If dominated, we STOP it early; otherwise we let it continue.
        """
        action = TrialScheduler.CONTINUE
        bracket = self.trial_info[trial.trial_id]

        # Ensure we have resource info + all objectives
        if self.time_attr not in result or not all(obj in result for obj in self.objectives):
            return action

        # E.g., resource = "training_iteration"
        resource = result[self.time_attr]

        # Gather raw objective values in order of self.objectives keys
        # Multiply by sign_vector, so bigger=better in each dimension
        metrics = np.array([result[obj] for obj in self.objectives]) * self.sign_vector

        # Let the bracket decide if we STOP or CONTINUE
        action = bracket.on_result(trial, resource, metrics)
        return action

    def on_trial_complete(self, tune_controller, trial, result):
        """
        Called when a trial finishes (either by STOP or by completing all epochs).
        We do one last rung check in case we haven't recorded it, then remove it.
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
        Called when a trial is removed (canceled externally).
        """
        if trial.trial_id in self.trial_info:
            del self.trial_info[trial.trial_id]

    def debug_string(self):
        """
        Useful for logging the bracket rung info, e.g. how many trials each rung has recorded.
        """
        return "\n".join([br.debug_string() for br in self.brackets])


class MultiObjectiveBracket:
    """
    A bracket for multi-objective ASHA. Each bracket has rung_levels like [1,3,9,...].
    For each rung we store a dict: trial_id -> metrics.

    When a trial hits rung i:
    1) We gather all rung i's existing solutions
    2) Filter out dominated solutions
    3) Reorder them by EPS_NET or NSGA_II
    4) Check if the new trial is dominated => STOP or CONTINUE
    """

    def __init__(self, rung_levels, strategy, sign_vector):
        self.rung_levels = rung_levels     # e.g. [1,3,9,27]
        self.strategy = strategy          # "eps_net" or "nsga_ii"
        self.sign_vector = sign_vector    # not strictly needed here, but kept for consistency

        # rungs is a list of dicts: each dict has "level" (the resource milestone)
        # and "recorded" = { trial_id -> metrics_array }
        # e.g.: [ {"level":1,"recorded":{}}, {"level":3,"recorded":{}}, {"level":9,"recorded":{}} ]
        self.rungs = [{"level": lvl, "recorded": {}} for lvl in rung_levels]
        print(f"Rungs: {self.rungs}")

    def on_result(self, trial, resource, metrics):
        """
        Called whenever a trial reports a result with 'resource' >= rung_level.
        We record the trial's metrics in that rung, then check if it's dominated.
        If dominated => STOP, else CONTINUE.
        """
        action = TrialScheduler.CONTINUE

        time_start = time.time()
        # We iterate over rung levels in ascending order
        for rung in self.rungs:
            # If trial hasn't reached this rung's resource, skip
            if resource < rung["level"]:
                continue

            # rung["recorded"] is the dict for that rung
            recorded = rung["recorded"]

            # If we've already recorded this trial in this rung, skip (no repeated record)
            if trial.trial_id in recorded:
                continue

            # Filter existing rung solutions, remove dominated, reorder, etc.
            candidates = self._calculate_candidates(recorded)

            # Check if new trial is dominated by the rung's best solutions
            if candidates is not None and not self._is_promotable(metrics, candidates):
                print(f"[Rung={rung['level']}] Trial {trial.trial_id} is dominated. -> STOP")
                action = TrialScheduler.STOP

            # Record this trial in rung
            recorded[trial.trial_id] = metrics
            # Only record in the first rung for which resource >= rung["level"], then break
            break

        time_end = time.time()
        
        print(f"Time taken for on_result: {time_end - time_start}")
        return action

    def _calculate_candidates(self, recorded):
        """
        From rung["recorded"], produce the set of best solutions:
          1) Convert recorded.values() => NxD array
          2) Filter out dominated
          3) Reorder them (EPS_NET or NSGA_II)
          4) Return that NxD array (in new order)
        """
        if not recorded:
            return None

        # points is a 2D array of shape (#trials_in_rung, #objectives)
        points = np.array(list(recorded.values()))

        # Step 1: Identify the subset of non-dominated points
        nd_indices = self._non_dominated_indices(points)
        nd_points = points[nd_indices]

        # Step 2: Reorder them via EpsNet or NSGA-II
        if self.strategy == EPS_NET:
            return self._eps_net(nd_points)
        elif self.strategy == NSGA_II:
            return self._nsga_ii(nd_points)

    def _is_promotable(self, new_metrics, candidates):
        """
        Returns True if new_metrics is NOT dominated by any row in 'candidates'.
        We say row c in 'candidates' dominates new_metrics if:
          c >= new_metrics in all dims AND c > new_metrics in at least 1 dim
        (since bigger=better).
        """
        for c in candidates:
            # c >= new_metrics in all dims
            # and at least one dimension is strictly greater
            if np.all(c >= new_metrics) and np.any(c > new_metrics):
                return False  # new_metrics is dominated => not promotable
        return True  # not dominated => promote

    def _non_dominated_indices(self, points):
        """
        Return indices of all non-dominated points in 'points'.
        After sign flipping, 'bigger is better' in each dimension.
        So p[i] dominates p[j] if p[i] >= p[j] in every dimension,
        and p[i] > p[j] in at least one dimension.

        This is a standard 'Pareto filter': we eliminate all p[j]
        that are strictly worse than some p[i].
        """
        n = len(points)
        mask = np.ones(n, dtype=bool)  # True => keep, unless found dominated
        for i in range(n):
            # If p[i] was already found dominated, skip
            if not mask[i]:
                continue
            for j in range(n):
                if i == j or not mask[j]:
                    continue
                # If points[i] dominates points[j], remove j
                if (np.all(points[i] >= points[j]) and
                    np.any(points[i] > points[j])):
                    mask[j] = False
        # Return the indices of all True in 'mask'
        return np.where(mask)[0]

    # ----------------------------------------------------------------
    # EpsNet + NSGA-II for reordering the non-dominated subset
    # ----------------------------------------------------------------
    def _eps_net(self, points):
        """
        Farthest-first approach:
          1) Start with index 0 as selected
          2) Iteratively pick the point that has the largest min-distance
             to the already selected set, ensuring coverage diversity.
        """
        if len(points) == 0:
            return None

        selected = [0]  # start with 0
        remaining = list(range(1, len(points)))

        while remaining:
            # For each candidate i in remaining, compute distance to nearest selected
            distances = []
            for i in remaining:
                # Dist from points[i] to the closest among the 'selected'
                min_dist = np.min([np.linalg.norm(points[i] - points[j]) for j in selected])
                distances.append(min_dist)

            # The farthest candidate from any selected is appended
            idx = np.argmax(distances)
            chosen_index = remaining.pop(idx)
            selected.append(chosen_index)

        # Return them in the farthest-first order
        return points[selected]

    def _nsga_ii(self, points):
        """
        Non-dominated sorting => front layers => within each front, sort by crowding distance.
        Return all points in the order of front+crowding.
        """
        if len(points) == 0:
            return None

        # Step 1: get fronts
        fronts = self._fast_nondominated_sort(points)
        result_indices = []
        for front in fronts:
            # Step 2: crowding distance for that front
            cdist = self._crowding_distance(points[front])
            # Step 3: sort front in descending order of cdist => prefer more spread out
            sorted_front = sorted(front, key=lambda x: -cdist[x])
            result_indices.extend(sorted_front)

        return points[result_indices]

    def _fast_nondominated_sort(self, points):
        """
        Identify front layers:
          - front 1 = points with no one dominating them
          - front 2 = points only dominated by front 1
          - ...
        """
        n = len(points)
        domination_counts = np.zeros(n, dtype=int)   # how many dominate me?
        dominated = [[] for _ in range(n)]           # who I dominate

        # Build domination relationships
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                # i dominated by j if j >= i and j>i in at least 1 dim
                if (np.all(points[j] >= points[i]) and
                    np.any(points[j] > points[i])):
                    domination_counts[i] += 1
                    dominated[j].append(i)

        # front[0] = those with 0 dominators
        fronts = [[]]
        for i in range(n):
            if domination_counts[i] == 0:
                fronts[0].append(i)

        # BFS-like expansion: once we pick front k, reduce dom_counts of those they dominate
        i = 0
        while len(fronts[i]) > 0:
            next_front = []
            for idx in fronts[i]:
                for d_idx in dominated[idx]:
                    domination_counts[d_idx] -= 1
                    if domination_counts[d_idx] == 0:
                        next_front.append(d_idx)
            i += 1
            fronts.append(next_front)

        return fronts[:-1]  # last front is empty

    def _crowding_distance(self, points):
        """
        For each front, we compute a crowding distance that measures how isolated each point is.
        The front is then sorted by descending distance => more spread out solutions come first.
        """
        n = len(points)
        if n == 0:
            return []

        dists = np.zeros(n)
        num_dims = points.shape[1]

        for dim in range(num_dims):
            # Sort by ascending values in this dim
            sorted_idx = np.argsort(points[:, dim])
            # The boundary points get infinite crowding distance
            dists[sorted_idx[0]] = np.inf
            dists[sorted_idx[-1]] = np.inf

            min_val = points[sorted_idx[0], dim]
            max_val = points[sorted_idx[-1], dim]
            diff = max_val - min_val

            if diff == 0:
                # All points same in this dimension => crowding dist doesn't change
                continue

            # For interior points, measure how large the 'gap' is
            for rank_idx in range(1, n - 1):
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

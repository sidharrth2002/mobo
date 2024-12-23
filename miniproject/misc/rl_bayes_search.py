from typing import Dict, List, Optional
from ray.tune.search import Searcher
from ray.rllib.algorithms.ppo import PPO, PPOConfig
import numpy as np
import bayesian_optimization as byo
from collections import defaultdict


class RLBayesOptSearch(Searcher):
    """RL-Augmented Bayesian Optimization Searcher."""

    def __init__(
        self,
        space: Optional[Dict] = None,
        metric: Optional[str] = None,
        mode: Optional[str] = None,
        points_to_evaluate: Optional[List[Dict]] = None,
        utility_kwargs: Optional[Dict] = None,
        random_state: int = 42,
        random_search_steps: int = 10,
        verbose: int = 0,
        patience: int = 5,
        skip_duplicate: bool = True,
    ):
        super().__init__(metric=metric, mode=mode)

        # Bayesian Optimization Setup
        self._config_counter = defaultdict(int)
        self._patience = patience
        self.repeat_float_precision = 5
        self._skip_duplicate = skip_duplicate

        if utility_kwargs is None:
            utility_kwargs = dict(kind="ucb", kappa=2.576, xi=0.0)

        self.utility_kwargs = utility_kwargs
        self._metric_op = 1.0 if mode == "max" else -1.0

        self.optimizer = None
        if space:
            self._space = self.convert_search_space(space)
            self.optimizer = byo.BayesianOptimization(
                f=None,
                pbounds=self._space,
                verbose=verbose,
                random_state=random_state,
            )
        self.points_to_evaluate = points_to_evaluate or []
        self.random_search_trials = random_search_steps
        self._total_random_search_trials = 0
        self._buffered_trial_results = []

        # RL Agent Setup
        self.rl_config = (
            PPOConfig()
            .environment(env=self.create_rl_env())
            .framework("torch")
            .rollouts(num_env_runners=1)
            .training(model={"fcnet_hiddens": [128, 128]})
        )
        self.rl_agent = self.rl_config.build()

    def create_rl_env(self):
        """Custom RL environment for tuning acquisition function parameters."""
        import gymnasium as gym
        from gymnasium import spaces

        class AcquisitionEnv(gym.Env):
            def __init__(self, searcher):
                self.searcher = searcher
                self.action_space = spaces.Box(
                    low=np.array([0.5]), high=np.array([10.0]), shape=(1,), dtype=np.float32
                )
                self.observation_space = spaces.Box(
                    low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32
                )

            def reset(self):
                return np.array([0.0, 1.0]), {}

            def step(self, action):
                kappa = action[0]
                self.searcher.utility_kwargs["kappa"] = kappa
                reward = -self.searcher.evaluate_acquisition_performance()
                return np.array([0.0, 1.0]), reward, True, {}

        return AcquisitionEnv(self)

    def suggest(self, trial_id: str) -> Optional[Dict]:
        """Suggest the next configuration."""
        if not self.optimizer:
            raise RuntimeError("Optimizer is not set up correctly.")

        if self.points_to_evaluate:
            config = self.points_to_evaluate.pop(0)
        else:
            # Use RL-augmented acquisition function to suggest next point
            kappa = self.rl_agent.compute_single_action([0.0, 1.0])[0]
            self.utility_kwargs["kappa"] = kappa
            utility_function = byo.UtilityFunction(**self.utility_kwargs)
            config = self.optimizer.suggest(utility_function)

        config_hash = _dict_hash(config, self.repeat_float_precision)
        already_seen = config_hash in self._config_counter
        self._config_counter[config_hash] += 1

        if already_seen and self._skip_duplicate:
            return None

        self._live_trial_mapping[trial_id] = config
        return config

    def on_trial_complete(self, trial_id: str, result: Optional[Dict] = None, error: bool = False):
        """Handle completed trial."""
        params = self._live_trial_mapping.pop(trial_id, None)
        if result is None or params is None or error:
            return

        if len(self._buffered_trial_results) >= self.random_search_trials:
            self._register_result(params, result)
        else:
            self._buffered_trial_results.append((params, result))
            if len(self._buffered_trial_results) == self.random_search_trials:
                for params, result in self._buffered_trial_results:
                    self._register_result(params, result)
                self._buffered_trial_results.clear()

        # Train the RL agent after processing trial results
        self.rl_agent.train()

    def _register_result(self, params: Dict, result: Dict):
        """Register the trial result with the optimizer."""
        if result[self.metric] is None:
            return
        self.optimizer.register(params, self._metric_op * result[self.metric])

    def evaluate_acquisition_performance(self):
        """Evaluate acquisition function performance."""
        utility_function = byo.UtilityFunction(**self.utility_kwargs)
        suggested = self.optimizer.suggest(utility_function)
        # Assume performance is derived from internal metrics (e.g., accuracy, loss)
        return self.optimizer.space.max()["target"]  # Placeholder logic

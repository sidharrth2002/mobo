from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties
from ray import tune, air
from ray.air import session
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.ax import AxSearch

def evaluate(parameter: dict, checkpoint_dir=None):
    session.report(
        {
            "a": parameter["a"],
            "b": parameter["b"],
        }
    )

ax_client = AxClient(
    verbose_logging=False,
    enforce_sequential_optimization=False,
)
ax_client.create_experiment(
    name="test",
    parameters=[
        {
            "name": "a",
            "type": "range",
            "value_type": "float",
            "bounds": [0, 1.0],
        },
        {
            "name": "b",
            "type": "range",
            "value_type": "float",
            "bounds": [0, 1.0],
        },
    ],
    objectives={
        "a": ObjectiveProperties(minimize=True, threshold=0.5),
        "b": ObjectiveProperties(minimize=True, threshold=0.5),
    },
    overwrite_existing_experiment=True,
    is_test=False,
)

algo = AxSearch(ax_client=ax_client)
algo = ConcurrencyLimiter(algo, max_concurrent=4)
tuner = tune.Tuner(
    tune.with_resources(evaluate, resources={"cpu": 1}),
    tune_config=tune.TuneConfig(search_alg=algo, num_samples=60),
    run_config=air.RunConfig(
        local_dir="./test_ray",
        verbose=0,
    ),
)
tuner.fit()
import ray
from ray import tune
from ray.tune.search.ax import AxSearch
from ax.service.ax_client import AxClient

def trainable(config):
    # Example objectives: minimize loss and maximize accuracy
    x = config["x"]
    y = config["y"]
    loss = (x - 1)**2 + (y - 1)**2  # Example: a simple quadratic loss
    accuracy = -loss + 2           # Inverse relation for demonstration
    
    tune.report(loss=loss, accuracy=accuracy)

# Initialize AxClient for multi-objective optimization
ax_client = AxClient()
ax_client.create_experiment(
    name="multi_objective_experiment",
    parameters=[
        {"name": "x", "type": "range", "bounds": [-5.0, 5.0]},
        {"name": "y", "type": "range", "bounds": [-5.0, 5.0]},
    ],
    objective_name="loss",
    minimize=True,
    outcome_constraints=["accuracy >= -1.0"],  # Example constraint
)

# Convert Ax experiment into a Ray Tune-compatible search algorithm
ax_search = AxSearch(ax_client)

# Run the optimization
analysis = tune.run(
    trainable,
    search_alg=ax_search,
    num_samples=20,
    metric="loss",
    mode="min",
)

# Best results
print("Best config: ", analysis.best_config)
print("Best trial results: ", analysis.best_result)

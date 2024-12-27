import os
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.ax import AxSearch
from ray import air
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

###################################
# Step 1: Initialize Ray locally
###################################
# Initialize Ray on the local machine with a few CPUs (adjust as needed)
ray.init(num_cpus=6, num_gpus=2, include_dashboard=True)

print(ray.cluster_resources())

###################################
# Step 2: Create a Toy Dataset
###################################
# For demonstration, we'll create a synthetic dataset of random inputs
# and a binary classification target. In real scenarios, load your actual data.
def get_data():
    X = torch.randn(1000, 20)  # 1000 samples, 20 features
    # Let's say we have a binary classification: 0 or 1
    y = (X[:, 0] > 0).long()  # Some arbitrary target for demonstration
    dataset = TensorDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    return train_loader

###################################
# Step 3: Define the Model Creation and Training
###################################
def create_model(num_layers, hidden_units, input_dim=20, output_dim=2):
    # Dynamically create a simple feed-forward architecture
    layers = []
    in_dim = input_dim
    for i in range(num_layers):
        layers.append(nn.Linear(in_dim, hidden_units))
        layers.append(nn.ReLU())
        in_dim = hidden_units
    layers.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*layers)

def train_model(config):
    # config will contain hyperparameters and architecture parameters
    # e.g., {"num_layers": 2, "hidden_units": 64, "lr": 0.001, ...}
    num_layers = config["num_layers"]
    hidden_units = config["hidden_units"]
    lr = config["lr"]
    epochs = 5  # For demonstration, keep it small

    train_loader = get_data()
    model = create_model(num_layers, hidden_units)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_x.size(0)
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

        avg_loss = running_loss / total
        accuracy = correct / total
        
        # Ray Tune communicates metrics by tune.report
        ray.train.report(dict(loss=avg_loss, accuracy=accuracy))

###################################
# Step 4: Define the Parameter Space and Ax Search
###################################
# We'll search over:
# - num_layers: discrete choice [1, 2, 3]
# - hidden_units: discrete choice [32, 64, 128]
# - learning rate: continuous uniform from 1e-4 to 1e-2
search_space = {
    "num_layers": tune.choice([1, 2, 3]),
    "hidden_units": tune.choice([32, 64, 128]),
    "lr": tune.loguniform(1e-4, 1e-2)
}

# Create an AxSearch object. We specify the objective metric and goal.
ax_search = AxSearch(
    metric="accuracy",
    mode="max"
)

###################################
# Step 5: Define a Scheduler (optional)
###################################
# We'll use ASHAScheduler to early-stop poorly performing trials.
scheduler = ASHAScheduler(
    # metric="accuracy",
    # mode="max",
    max_t=5,  # max epochs
    grace_period=1,
    reduction_factor=2
)

###################################
# Step 6: Run Tuner
###################################
tuner = tune.Tuner(
    tune.with_resources(
        train_model, resources={"cpu": 1}
    ),
    param_space=search_space,
    tune_config=tune.TuneConfig(
        metric="accuracy",
        mode="max",
        search_alg=ax_search,
        scheduler=scheduler,
        num_samples=1000,  # total number of trials to run,
        max_concurrent_trials=6,  # number of trials to run concurrently
    ),
    run_config=air.RunConfig(
        storage_path=os.path.abspath("nas_with_ax"),
        name="nas_with_ax",
    )
)

results = tuner.fit()

###################################
# Step 7: Review Results
###################################
best_result = results.get_best_result(metric="accuracy", mode="max")
print("Best trial config:", best_result.config)
print("Best trial final accuracy:", best_result.metrics["accuracy"])

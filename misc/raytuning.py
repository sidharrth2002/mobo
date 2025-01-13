import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from ray import tune
# from ray.rllib.agents.ppo import PPOTrainer

# Define a simple model for demonstration (e.g., LeNet for MNIST)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Training function with model pruning
def train_model(config):
    pruning_percentage = config["pruning_percentage"]
    learning_rate = config["learning_rate"]
    retraining_epochs = config["retraining_epochs"]

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_data = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # for name, module in model.named_modules():
    #     if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
    #         torch.nn.utils.prune.l1_unstructured(module, name='weight', amount=pruning_percentage)

    for epoch in range(retraining_epochs):
        model.train()
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    accuracy = evaluate_model(model)
    compression_ratio = calculate_compression_ratio(model)

    tune.report(accuracy=accuracy, compression_ratio=compression_ratio, reward=accuracy - compression_ratio * 0.1)

def evaluate_model(model):
    test_data = datasets.MNIST(root="data", train=False, download=True, transform=transforms.ToTensor())
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return correct / total

def calculate_compression_ratio(model):
    total_params = sum(p.numel() for p in model.parameters())
    pruned_params = sum(p.numel() for p in model.parameters() if hasattr(p, "weight_mask"))
    return (total_params - pruned_params) / total_params

# optimization
def run_search():
    # Define the search space
    search_space = {
        "pruning_percentage": tune.uniform(0.1, 0.8),
        "learning_rate": tune.loguniform(1e-4, 1e-2),
        "retraining_epochs": tune.randint(1, 10),
    }

    analysis = tune.run(
        train_model,
        config=search_space,
        metric="reward",
        mode="max",
        num_samples=20,
    )

    print("Best hyperparameters found were: ", analysis.best_config)

if __name__ == "__main__":
    run_search()
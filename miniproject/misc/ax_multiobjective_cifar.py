"""
More advanced Neural Architecture Search
Explore architectures such as ResNet, MobileNet, etc.
"""

import argparse
from dataclasses import dataclass
import tempfile
import uuid
from filelock import FileLock
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
)
from torchmetrics import Accuracy
from lib.axsearch_multiobjective import AxSearchMultiObjective
from ax.service.ax_client import AxClient
from ray.tune.search import ConcurrencyLimiter
from ax.service.utils.instantiation import ObjectiveProperties
import logging
from ray.train import RunConfig, ScalingConfig, CheckpointConfig
from ray.tune import TuneConfig, Tuner
from lib.ax_torchtrainer import TorchTrainerMultiObjective
from lib.mobo_asha_4 import MultiObjectiveAsyncHyperBandScheduler
import pickle
from ax.service.utils.report_utils import _pareto_frontier_scatter_2d_plotly
import json
import os
from ax.plot.contour import interact_contour_plotly

import signal
import sys
from ax.modelbridge.cross_validation import compute_diagnostics, cross_validate
from ax.plot.diagnostic import interact_cross_validation_plotly

import torch

# is CUDA available?
print(f"Is CUDA available: {torch.cuda.is_available()}")
print(f"Number of GPUs: {torch.cuda.device_count()}")
# sys.exit(0)


def handle_sigint(signum, frame):
    print("Signal received, terminating...")
    sys.exit(0)


signal.signal(signal.SIGINT, handle_sigint)


logging.basicConfig(level=logging.INFO)


# parameters we are tuning
@dataclass
class ModelConfig:
    learning_rate: float
    batch_size: int
    layer_1_size: int
    layer_2_size: int
    layer_3_size: int
    dropout: float
    
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(BasicBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.stride = stride
        
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
        
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = self.relu(out)
        
        return out

class ResNet(nn.Module):
    def __init__(self, num_blocks, conv1_channels, num_classes=10):
        super(ResNet, self).__init__()

        self.in_channels = 16
        self.conv1 = nn.Conv2d(3, conv1_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(conv1_channels)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(64, num_blocks[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)
        
    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class CIFAR10Classifier(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        
        self.accuracy = Accuracy(task="multiclass", num_classes=10, top_k=1)
        
        # This paper proposes a neural architecture search space using ResNet as a framework, with search objectives including parameters for convolution, pooling, fully connected layers, and connectivity of the residual network. In addition to recognition accuracy, this paper uses the loss value on the validation set as a secondary objective for optimization. The experimental results demonstrate that the search space of this paper together with the optimisation approach can find competitive network architectures on the MNIST, Fashion-MNIST and CIFAR100 datasets.

        # model parameters
        self.model = ResNet(num_blocks=[2, 2, 2], conv1_channels=config.conv1_channels, num_classes=10)

class MNISTClassifier(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.accuracy = Accuracy(task="multiclass", num_classes=10, top_k=1)

        # model parameters
        self.layer1 = nn.Linear(28 * 28, config.layer_1_size)
        self.layer2 = nn.Linear(config.layer_1_size, config.layer_2_size)
        self.layer3 = nn.Linear(config.layer_2_size, config.layer_3_size)
        self.layer4 = nn.Linear(config.layer_3_size, 10)
        self.dropout = nn.Dropout(config.dropout)

        # training parameters
        self.learning_rate = config.learning_rate

        self.eval_loss = []
        self.eval_accuracy = []

        self.model_params = int(round(sum(p.numel() for p in self.parameters())))
        print(f"Model parameters: {self.model_params}")

    def forward(self, x):
        batch_size, channels, width, height = x.size()

        x = x.view(batch_size, -1)

        x = F.relu(self.layer1(x))
        x = self.dropout(x)

        x = F.relu(self.layer2(x))
        x = self.dropout(x)

        x = F.relu(self.layer3(x))
        x = self.dropout(x)

        x = self.layer4(x)
        x = torch.log_softmax(x, dim=1)
        return x

    def cross_entropy_loss(self, logits, labels):
        """
        Apply NLL loss because softmax is applied in the forward function
        """
        return F.nll_loss(logits, labels)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        accuracy = self.accuracy(logits, y)

        self.log("ptl/train_loss", loss)
        self.log("ptl/train_accuracy", accuracy)

        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        accuracy = self.accuracy(logits, y)

        self.eval_loss.append(loss)
        self.eval_accuracy.append(accuracy)

        return {
            "val_loss": loss,
            "val_accuracy": accuracy,
            "model_params": self.model_params,
        }

    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.eval_loss).mean()
        avg_acc = torch.stack(self.eval_accuracy).mean()

        # flop_count = FlopCountAnalysis(self, (1, 1, 28, 28)).total()

        self.log("ptl/val_loss", avg_loss, sync_dist=True)
        self.log("ptl/val_accuracy", avg_acc, sync_dist=True)
        self.log("ptl/model_params", self.model_params, sync_dist=True)
        # self.log("ptl/flops", flop_count, sync_dist=True)
        self.eval_loss.clear()
        self.eval_accuracy.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=128, data_path="./data"):
        super().__init__()
        self.data_dir = data_path
        self.batch_size = batch_size
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

    def setup(self, stage=None):
        with FileLock(f"{self.data_dir}.lock"):
            if not os.path.exists(os.path.join(self.data_dir, "MNIST")):
                print("Downloading MNIST dataset...")
            else:
                print("Using cached MNIST dataset...")

            mnist = MNIST(
                self.data_dir, train=True, download=True, transform=self.transform
            )
            self.mnist_train, self.mnist_val = random_split(mnist, [55000, 5000])
            self.mnist_test = MNIST(
                self.data_dir, train=False, download=True, transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=4)


class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, batch_size=128, data_path="./data"):
        super().__init__()
        self.data_dir = data_path
        self.batch_size = batch_size
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

    def setup(self, stage=None):
        with FileLock(f"{self.data_dir}.lock"):
            if not os.path.exists(os.path.join(self.data_dir, "CIFAR10")):
                print("Downloading CIFAR10 dataset...")
            else:
                print("Using cached CIFAR10 dataset...")

            cifar = CIFAR10(
                self.data_dir, train=True, download=True, transform=self.transform
            )
            self.cifar_train, self.cifar_val = random_split(cifar, [45000, 5000])
            self.cifar_test = CIFAR10(
                self.data_dir, train=False, download=True, transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(self.cifar_train, batch_size=self.batch_size, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.cifar_val, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.cifar_test, batch_size=self.batch_size, num_workers=4)


def generate_uuid():
    """
    Generate a unique hash for the configuration
    """
    return uuid.uuid4().hex


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Multi-objective optimization")
    argparser.add_argument(
        "--num_samples", type=int, help="Number of samples to evaluate"
    )
    argparser.add_argument(
        "--max_num_epochs",
        type=int,
        help="Number of epochs to train the model under Torch Lightning",
    )
    argparser.add_argument("--num_gpus", type=int, help="Number of GPUs to use")
    # first objective is compulsory, other objectives are optional
    argparser.add_argument(
        "--objective_1", type=str, required=True, help="Objective 1 to optimize"
    )
    argparser.add_argument(
        "--objective_1_type",
        type=str,
        choices=["min", "max"],
        required=True,
        help="Type of optimization for objective 1 (min or max)",
    )
    argparser.add_argument(
        "--objective_1_threshold",
        type=float,
        required=True,
        help="Threshold for objective 1",
    )
    argparser.add_argument(
        "--objective_2", type=str, default=None, help="Objective 2 to optimize"
    )
    argparser.add_argument(
        "--objective_2_type",
        type=str,
        default=None,
        choices=["min", "max"],
        help="Type of optimization for objective 2 (min or max)",
    )
    argparser.add_argument(
        "--objective_2_threshold",
        type=float,
        default=None,
        help="Threshold for objective 2",
    )
    argparser.add_argument(
        "--objective_3", type=str, default=None, help="Objective 3 to optimize"
    )
    argparser.add_argument(
        "--objective_3_type",
        type=str,
        default=None,
        choices=["min", "max"],
        help="Type of optimization for objective 3 (min or max)",
    )
    argparser.add_argument(
        "--objective_3_threshold",
        type=float,
        default=None,
        help="Threshold for objective 3",
    )
    argparser.add_argument(
        "--max_concurrent", type=int, default=5, help="Max number of concurrent trials"
    )
    argparser.add_argument(
        "--use_scheduler",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
        help="If set to True, uses the MO-ASHA scheduler",
    )
    argparser.add_argument(
        "--scheduler_max_t",
        type=int,
        default=2,
        help="Maximum number of trials after which scheduler decides whether to stop",
    )
    argparser.add_argument(
        "--scheduler_grace_period",
        type=int,
        default=1,
        help="Grace period for the scheduler",
    )
    argparser.add_argument(
        "--scheduler_reduction_factor",
        type=int,
        default=4,
        help="Reduction factor for the scheduler",
    )
    argparser.add_argument(
        "--accelerator",
        type=str,
        choices=["auto", "cpu", "gpu", "mps"],
        default="auto",
    )
    argparser.add_argument(
        "--use_scaling_config",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
        help="If set to True, uses the scaling config",
    )
    argparser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Persistent location for storing the MNIST dataset",
    )
    argparser.add_argument(
        "--remark",
        type=str,
        default="",
        help="Note to save with the results to identify this run later",
    )

    args = argparser.parse_args()

    print(f"Running optimisation with args: {args}")

    ax_client = AxClient(verbose_logging=False)

    # to pass to ax client
    ax_client_objectives = {}
    # to pass to scheduler
    general_objectives = {}

    if args.objective_1 and args.objective_1_type:
        ax_client_objectives[args.objective_1] = ObjectiveProperties(
            minimize=args.objective_1_type == "min",
        )
        if args.objective_1_threshold:
            ax_client_objectives[args.objective_1].threshold = (
                args.objective_1_threshold
            )

        general_objectives[args.objective_1] = args.objective_1_type

    if args.objective_2 and args.objective_2_type:
        ax_client_objectives[args.objective_2] = ObjectiveProperties(
            minimize=args.objective_2_type == "min"
        )
        if args.objective_2_threshold:
            ax_client_objectives[args.objective_2].threshold = (
                args.objective_2_threshold
            )

        general_objectives[args.objective_2] = args.objective_2_type

    if args.objective_3 and args.objective_3_type:
        ax_client_objectives[args.objective_3] = ObjectiveProperties(
            minimize=args.objective_3_type == "min"
        )
        if args.objective_3_threshold:
            ax_client_objectives[args.objective_3].threshold = (
                args.objective_3_threshold
            )
        general_objectives[args.objective_3] = args.objective_3_type

    print(f"Using {len(ax_client_objectives)} Objectives: {ax_client_objectives}")

    ax_client.create_experiment(
        name="mnist_nas_multiobjective",
        parameters=[
            {"name": "layer_1_size", "type": "choice", "values": [16, 32]},
            {"name": "layer_2_size", "type": "choice", "values": [32, 64]},
            {"name": "layer_3_size", "type": "choice", "values": [64, 128]},
            {"name": "dropout", "type": "range", "bounds": [0.1, 0.3]},
            {"name": "batch_size", "type": "choice", "values": [64, 128]},
            {"name": "learning_rate", "type": "range", "bounds": [1e-4, 1e-1]},
        ],
        objectives=ax_client_objectives,
    )

    algo = AxSearchMultiObjective(ax_client=ax_client)
    # limit the number of concurrent trials
    print(f"Limiting concurrent trials to {args.max_concurrent}")
    algo = ConcurrencyLimiter(algo, max_concurrent=args.max_concurrent)

    run_config = RunConfig(
        checkpoint_config=CheckpointConfig(
            num_to_keep=2,
            checkpoint_score_attribute="ptl/val_accuracy",
            checkpoint_score_order="max",
        )
    )
    print(f"Run config: {run_config}")

    tune_config = TuneConfig(num_samples=args.num_samples, search_alg=algo)

    if args.use_scheduler:
        scheduler = MultiObjectiveAsyncHyperBandScheduler(
            max_t=args.scheduler_max_t,
            objectives=general_objectives,
            grace_period=args.scheduler_grace_period,
            reduction_factor=args.scheduler_reduction_factor,
        )

        tune_config.scheduler = scheduler

        print(f"Using MO-ASHA scheduler: {scheduler}")
    else:
        print("NOT USING ANY SCHEDULER, WOULD USE FIFO!")

    print(f"Tune config: {tune_config}")

    def train_func(config):
        data_module = MNISTDataModule(
            batch_size=config["batch_size"], data_path=args.data_path
        )
        print(f"printing config, {config}")

        # instantiate config object
        config = ModelConfig(
            learning_rate=config["learning_rate"],
            batch_size=config["batch_size"],
            layer_1_size=config["layer_1_size"],
            layer_2_size=config["layer_2_size"],
            layer_3_size=config["layer_3_size"],
            dropout=config["dropout"],
        )

        model = MNISTClassifier(config=config)

        print(f"Using accelerator: {args.accelerator}")

        trainer = pl.Trainer(
            devices="auto",
            accelerator=args.accelerator,
            strategy=RayDDPStrategy(),
            callbacks=[RayTrainReportCallback()],
            plugins=[RayLightningEnvironment()],
            enable_progress_bar=False,
            # TODO: put this back?
            max_epochs=args.max_num_epochs,
        )

        trainer = prepare_trainer(trainer)
        trainer.fit(model, datamodule=data_module)

    if args.use_scaling_config:
        # scaling_config = ScalingConfig(
        #     num_workers=min(5, torch.cuda.device_count()), use_gpu=True, resources_per_worker={"CPU": 5, "GPU": 1}
        # )

        # TODO: Idk how the scaling actually works, ignore for now
        scaling_config = ScalingConfig(
            num_workers=3, use_gpu=True, resources_per_worker={"CPU": 1, "GPU": 1}
        )

        print(f"Scaling config: {scaling_config}")

        ray_trainer = TorchTrainerMultiObjective(
            train_func,
            scaling_config=scaling_config,
            run_config=run_config,
        )
    else:
        ray_trainer = TorchTrainerMultiObjective(
            train_func,
            run_config=run_config,
        )

    tuner = Tuner(ray_trainer, tune_config=tune_config, run_config=run_config)

    tuning_results = tuner.fit()

    print(f"Results: {tuning_results}")

    configuration_hash = generate_uuid()

    # create a folder in results directory
    print(f"Creating a folder for configuration: {configuration_hash}")

    os.makedirs(f"results/{configuration_hash}", exist_ok=True)

    print(f"Configuration hash: {configuration_hash}")

    # save args used in this run
    with open(f"results/{configuration_hash}/args.json", "w") as f:
        json.dump(vars(args), f)

    # save ax client
    ax_client.save_to_json_file(f"results/{configuration_hash}/ax_client.json")

    # save tuning results
    tuning_results.get_dataframe().to_csv(
        f"results/{configuration_hash}/tuning_results_df.csv"
    )

    # pickle and save the tuning_results
    with open(f"results/{configuration_hash}/tuning_results.pkl", "wb") as f:
        print(
            f"Saving tuning results to file: results/{configuration_hash}/tuning_results.pkl"
        )
        pickle.dump(tuning_results, f)

    # save the pareto
    print(f"Plotting the pareto front for configuration: {configuration_hash}")
    pareto = _pareto_frontier_scatter_2d_plotly(ax_client.experiment)
    pareto.write_image(f"results/{configuration_hash}/pareto.png")
    pareto.write_html(f"results/{configuration_hash}/pareto.html")

    # save cv surrogate model
    # print(f"Saving CV surrogate model for configuration: {configuration_hash}")
    # cv = cross_validate(model=ax_client.generation_strategy.model)
    # compute_diagnostics(cv)
    # cv_plot = interact_cross_validation_plotly(cv)
    # cv_plot.write_image(f"results/{configuration_hash}/cv.png")
    # cv_plot.write_html(f"results/{configuration_hash}/cv.html")

    # save contour plot
    # print(f"Saving contour plot for configuration: {configuration_hash}")

    # for objective in general_objectives:
    #     countour_plot = interact_contour_plotly(
    #         model=ax_client.generation_strategy.model, metric_name=objective
    #     )
    #     countour_plot.write_image(
    #         f"results/{configuration_hash}/contour_{objective}.png"
    #     )
    #     countour_plot.write_html(
    #         f"results/{configuration_hash}/contour_{objective}.html"
    #     )

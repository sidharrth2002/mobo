This folder contains:

1. `ax_torchtrainer.py` - A subclassed Torch Trainer to be compatible with the way Ax handles parametrization, and pumps in parameters to the model. Essentially, it rebuilds the `train_loop_config` based on `kwargs` -- I had to build this because the existing `TorchTrainer` was incompatible with the way Ax passes hyperparameters to the Lightning Trainer. 

This folder contains two versions of the Multi-Objective ASHA scheduler:

1. `mobo_asha_naive` - Uses a naive Pareto front approximation to decide if a trial should be allowed to continue. It employs a stopping-based decision process.

2. `mobo_asha_epsnet` - Uses a Pareto front approximation as well, but contains additional functionality to sort the front using EpsNet and NSGA-ii. It was originally designed for a promotion-driven scheduling process, but I later found out that this paradigm is incompatible with the way Ray Tune does hyperband scheduling. 
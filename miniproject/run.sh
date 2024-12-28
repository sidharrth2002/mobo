#!/bin/bash
set -m

# runs without scheduler

CUDA_VISIBLE_DEVICES=2,4,5,6,7 python3 ax_multiobjective.py --num_samples 30 --max_num_epochs 4 --num_gpus 2 --objective_1 ptl/val_accuracy --objective_1_type max --objective_1_threshold 0.90 --objective_2 ptl/model_params --objective_2_threshold 100000 --objective_2_type min --max_concurrent 10 --accelerator gpu --data_path /home/sn666/large-scale-data-processing/miniproject/data --use_scaling_config

# CUDA_VISIBLE_DEVICES=0,1,2 python3 ax_multiobjective.py --num_samples 20 --max_num_epochs 3 --num_gpus 2 --objective_1 ptl/val_accuracy --objective_1_type max --objective_1_threshold 0.90 --objective_2 ptl/model_params --objective_2_threshold 100000 --objective_2_type min --max_concurrent 5 --use_scheduler --scheduler_max_t 3 --accelerator gpu --use_scaling_config

PGID=$$

# Trap SIGINT (Ctrl+C) and SIGTERM signals to kill all processes in the group
trap "echo 'Terminating all processes...'; kill -TERM -$PGID" SIGINT SIGTERM

# Wait for all background processes to finish
wait
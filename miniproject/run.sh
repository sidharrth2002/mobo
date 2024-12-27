#!/bin/bash
set -m

python3 ax_multiobjective.py --num_samples 2 --max_num_epochs 1 --num_gpus 2 --objective_1 ptl/val_accuracy --objective_1_type max --objective_1_threshold 0.90 --objective_2 ptl/model_params --objective_2_threshold 100000 --objective_2_type min --max_concurrent 5 --use_scheduler --scheduler_max_t 3 --accelerator cpu

PGID=$$

# Trap SIGINT (Ctrl+C) and SIGTERM signals to kill all processes in the group
trap "echo 'Terminating all processes...'; kill -TERM -$PGID" SIGINT SIGTERM

# Wait for all background processes to finish
wait
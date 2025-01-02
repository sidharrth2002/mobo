#!/bin/bash
set -m

# Generate a unique log file for each Python script invocation
generate_log_file() {
  local script_name=$(basename "$1" .py)
  local timestamp=$(date "+%Y%m%d-%H%M%S")
  echo "logs/${script_name}_${timestamp}.log"
}

# RUNS WITHOUT SCHEDULER
# # max accuracy and min params
# LOG_FILE=$(generate_log_file "ax_multiobjective.py")
# echo "Logging to $LOG_FILE"
# CUDA_VISIBLE_DEVICES=2,4,5,6,7 python3 ax_multiobjective.py --num_samples 25 --max_num_epochs 4 --objective_1 ptl/val_accuracy --objective_1_type max --objective_1_threshold 0.90 --objective_2 ptl/model_params --objective_2_threshold 100000 --objective_2_type min --max_concurrent 10 --accelerator gpu --data_path /home/sn666/large-scale-data-processing/miniproject/data --remark maxaccminparam --use_scaling_config| tee -a "$LOG_FILE"

# # max accuracy and min loss
# LOG_FILE=$(generate_log_file "ax_multiobjective.py")
# echo "Logging to $LOG_FILE"
# CUDA_VISIBLE_DEVICES=2,4,5,6,7 python3 ax_multiobjective.py --num_samples 25 --max_num_epochs 4 --objective_1 ptl/val_accuracy --objective_1_type max --objective_1_threshold 0.90 --objective_2 ptl/val_loss --objective_2_threshold 0.5 --objective_2_type min --max_concurrent 10 --accelerator gpu --data_path /home/sn666/large-scale-data-processing/miniproject/data --remark micaccminloss --use_scaling_config | tee -a "$LOG_FILE"

# # min loss and min params
# LOG_FILE=$(generate_log_file "ax_multiobjective.py")
# echo "Logging to $LOG_FILE"
# CUDA_VISIBLE_DEVICES=2,4,5,6,7 python3 ax_multiobjective.py --num_samples 25 --max_num_epochs 4 --objective_1 ptl/val_loss --objective_1_type min --objective_1_threshold 0.5 --objective_2 ptl/model_params --objective_2_threshold 100000 --objective_2_type min --max_concurrent 10 --accelerator gpu --data_path /home/sn666/large-scale-data-processing/miniproject/data --remark minlossminparam --use_scaling_config | tee -a "$LOG_FILE"

# # max accuracy, min loss and min params
# LOG_FILE=$(generate_log_file "ax_multiobjective.py")
# echo "Logging to $LOG_FILE"
# CUDA_VISIBLE_DEVICES=2,4,5,6,7 python3 ax_multiobjective.py --num_samples 25 --max_num_epochs 4 --objective_1 ptl/val_accuracy --objective_1_type max --objective_1_threshold 0.90 --objective_2 ptl/val_loss --objective_2_threshold 0.5 --objective_2_type min --objective_3 ptl/model_params --objective_3_threshold 100000 --objective_3_type min --max_concurrent 10 --accelerator gpu --data_path /home/sn666/large-scale-data-processing/miniproject/data --remark maxaccminlossminparam --use_scaling_config | tee -a "$LOG_FILE"

# RUNS WITH SCHEDULER

# max accuracy and min params
# LOG_FILE=$(generate_log_file "ax_multiobjective.py")
# echo "Logging to $LOG_FILE"
# CUDA_VISIBLE_DEVICES=2,4,5,6,7 python3 ax_multiobjective.py --num_samples 25 --max_num_epochs 4 --objective_1 ptl/val_accuracy --objective_1_type max --objective_1_threshold 0.90 --objective_2 ptl/model_params --objective_2_threshold 100000 --objective_2_type min --max_concurrent 10 --accelerator gpu --data_path /home/sn666/large-scale-data-processing/miniproject/data --use_scheduler --remark moasha/maxaccminparam --use_scaling_config | tee -a "$LOG_FILE"

# # max accuracy and min loss
# LOG_FILE=$(generate_log_file "ax_multiobjective.py")
# echo "Logging to $LOG_FILE"
# CUDA_VISIBLE_DEVICES=2,4,5,6,7 python3 ax_multiobjective.py --num_samples 25 --max_num_epochs 4 --objective_1 ptl/val_accuracy --objective_1_type max --objective_1_threshold 0.90 --objective_2 ptl/val_loss --objective_2_threshold 0.5 --objective_2_type min --max_concurrent 10 --accelerator gpu --data_path /home/sn666/large-scale-data-processing/miniproject/data --use_scheduler --remark moasha/maxaccminloss --use_scaling_config | tee -a "$LOG_FILE"

# # min loss and min params
# LOG_FILE=$(generate_log_file "ax_multiobjective.py")
# echo "Logging to $LOG_FILE"
# CUDA_VISIBLE_DEVICES=2,4,5,6,7 python3 ax_multiobjective.py --num_samples 25 --max_num_epochs 4 --objective_1 ptl/val_loss --objective_1_type min --objective_1_threshold 0.5 --objective_2 ptl/model_params --objective_2_threshold 100000 --objective_2_type min --max_concurrent 10 --accelerator gpu --data_path /home/sn666/large-scale-data-processing/miniproject/data --use_scheduler --remark moasha/minlossminparam --use_scaling_config | tee -a "$LOG_FILE"

# # max accuracy, min loss and min params
# LOG_FILE=$(generate_log_file "ax_multiobjective.py")
# echo "Logging to $LOG_FILE"
# CUDA_VISIBLE_DEVICES=2,4,5,6,7 python3 ax_multiobjective.py --num_samples 25 --max_num_epochs 4 --objective_1 ptl/val_accuracy --objective_1_type max --objective_1_threshold 0.90 --objective_2 ptl/val_loss --objective_2_threshold 0.5 --objective_2_type min --objective_3 ptl/model_params --objective_3_threshold 100000 --objective_3_type min --max_concurrent 10 --accelerator gpu --data_path /home/sn666/large-scale-data-processing/miniproject/data --use_scheduler --remark moasha/maxaccminlossminparam --use_scaling_config | tee -a "$LOG_FILE"


# max_t = 3

# # max accuracy and min params
# LOG_FILE=$(generate_log_file "ax_multiobjective.py")
# echo "Logging to $LOG_FILE"
# CUDA_VISIBLE_DEVICES=2,4,5,6,7 python3 ax_multiobjective.py --num_samples 25 --max_num_epochs 4 --objective_1 ptl/val_accuracy --objective_1_type max --objective_1_threshold 0.90 --objective_2 ptl/model_params --objective_2_threshold 100000 --objective_2_type min --max_concurrent 10 --accelerator gpu --data_path /home/sn666/large-scale-data-processing/miniproject/data --use_scheduler --scheduler_max_t 3 --remark moasha/maxt3/maxaccminparam --use_scaling_config | tee -a "$LOG_FILE"

# # max accuracy and min loss
# LOG_FILE=$(generate_log_file "ax_multiobjective.py")
# echo "Logging to $LOG_FILE"
# CUDA_VISIBLE_DEVICES=2,4,5,6,7 python3 ax_multiobjective.py --num_samples 25 --max_num_epochs 4 --objective_1 ptl/val_accuracy --objective_1_type max --objective_1_threshold 0.90 --objective_2 ptl/val_loss --objective_2_threshold 0.5 --objective_2_type min --max_concurrent 10 --accelerator gpu --data_path /home/sn666/large-scale-data-processing/miniproject/data --use_scheduler --scheduler_max_t 3 --remark moasha/maxt3/maxaccminloss --use_scaling_config | tee -a "$LOG_FILE"

# # min loss and min params
# LOG_FILE=$(generate_log_file "ax_multiobjective.py")
# echo "Logging to $LOG_FILE"
# CUDA_VISIBLE_DEVICES=2,4,5,6,7 python3 ax_multiobjective.py --num_samples 25 --max_num_epochs 4 --objective_1 ptl/val_loss --objective_1_type min --objective_1_threshold 0.5 --objective_2 ptl/model_params --objective_2_threshold 100000 --objective_2_type min --max_concurrent 10 --accelerator gpu --data_path /home/sn666/large-scale-data-processing/miniproject/data --use_scheduler --scheduler_max_t 3 --remark moasha/maxt3/minlossminparam --use_scaling_config | tee -a "$LOG_FILE"

# # max accuracy, min loss and min params
# LOG_FILE=$(generate_log_file "ax_multiobjective.py")
# echo "Logging to $LOG_FILE"
# CUDA_VISIBLE_DEVICES=2,4,5,6,7 python3 ax_multiobjective.py --num_samples 25 --max_num_epochs 4 --objective_1 ptl/val_accuracy --objective_1_type max --objective_1_threshold 0.90 --objective_2 ptl/val_loss --objective_2_threshold 0.5 --objective_2_type min --objective_3 ptl/model_params --objective_3_threshold 100000 --objective_3_type min --max_concurrent 10 --accelerator gpu --data_path /home/sn666/large-scale-data-processing/miniproject/data --use_scheduler --scheduler_max_t 3 --remark moasha/maxt3/maxaccminlossminparam --use_scaling_config | tee -a "$LOG_FILE"

# with different reduction factor

# max accuracy and min params
# LOG_FILE=$(generate_log_file "ax_multiobjective.py")
# echo "Logging to $LOG_FILE"
# CUDA_VISIBLE_DEVICES=2,4,5,6,7 python3 ax_multiobjective.py --num_samples 25 --max_num_epochs 4 --objective_1 ptl/val_accuracy --objective_1_type max --objective_1_threshold 0.90 --objective_2 ptl/model_params --objective_2_threshold 100000 --objective_2_type min --max_concurrent 10 --accelerator gpu --data_path /home/sn666/large-scale-data-processing/miniproject/data --use_scheduler --scheduler_max_t 4 --scheduler_grace_period 1 --scheduler_reduction_factor 3 --remark moasha/maxt4red3/maxaccminparam --use_scaling_config | tee -a "$LOG_FILE"

# # max accuracy and min loss
# LOG_FILE=$(generate_log_file "ax_multiobjective.py")
# echo "Logging to $LOG_FILE"
# CUDA_VISIBLE_DEVICES=2,4,5,6,7 python3 ax_multiobjective.py --num_samples 25 --max_num_epochs 4 --objective_1 ptl/val_accuracy --objective_1_type max --objective_1_threshold 0.90 --objective_2 ptl/val_loss --objective_2_threshold 0.5 --objective_2_type min --max_concurrent 10 --accelerator gpu --data_path /home/sn666/large-scale-data-processing/miniproject/data --use_scheduler --scheduler_max_t 4 --scheduler_grace_period 1 --scheduler_reduction_factor 3 --remark moasha/maxt4red3/maxaccminloss --use_scaling_config | tee -a "$LOG_FILE"

# # min loss and min params
# LOG_FILE=$(generate_log_file "ax_multiobjective.py")
# echo "Logging to $LOG_FILE"
# CUDA_VISIBLE_DEVICES=2,4,5,6,7 python3 ax_multiobjective.py --num_samples 25 --max_num_epochs 4 --objective_1 ptl/val_loss --objective_1_type min --objective_1_threshold 0.5 --objective_2 ptl/model_params --objective_2_threshold 100000 --objective_2_type min --max_concurrent 10 --accelerator gpu --data_path /home/sn666/large-scale-data-processing/miniproject/data --use_scheduler --scheduler_max_t 4 --scheduler_grace_period 1 --scheduler_reduction_factor 3 --remark moasha/maxt4red3/minlossminparam --use_scaling_config | tee -a "$LOG_FILE"

# # max accuracy, min loss and min params
# LOG_FILE=$(generate_log_file "ax_multiobjective.py")
# echo "Logging to $LOG_FILE"
# CUDA_VISIBLE_DEVICES=2,4,5,6,7 python3 ax_multiobjective.py --num_samples 25 --max_num_epochs 4 --objective_1 ptl/val_accuracy --objective_1_type max --objective_1_threshold 0.90 --objective_2 ptl/val_loss --objective_2_threshold 0.5 --objective_2_type min --objective_3 ptl/model_params --objective_3_threshold 100000 --objective_3_type min --max_concurrent 10 --accelerator gpu --data_path /home/sn666/large-scale-data-processing/miniproject/data --use_scheduler ---scheduler_max_t 4 --scheduler_grace_period 1 --scheduler_reduction_factor 3 --remark moasha/maxt4red3/maxaccminlossminparam --use_scaling_config | tee -a "$LOG_FILE"


# CUDA_VISIBLE_DEVICES=0,1,2 python3 ax_multiobjective.py --num_samples 20 --max_num_epochs 3 --num_gpus 2 --objective_1 ptl/val_accuracy --objective_1_type max --objective_1_threshold 0.90 --objective_2 ptl/model_params --objective_2_threshold 100000 --objective_2_type min --max_concurrent 5 --use_scheduler --scheduler_max_t 3 --accelerator gpu --use_scaling_config | tee -a "$LOG_FILE"

# max accuracy and min params
# LOG_FILE=$(generate_log_file "ax_multiobjective.py")
# echo "Logging to $LOG_FILE"
# CUDA_VISIBLE_DEVICES=2,4,5,6,7 python3 ax_multiobjective.py --num_samples 25 --max_num_epochs 4 --objective_1 ptl/val_accuracy --objective_1_type max --objective_1_threshold 0.90 --objective_2 ptl/model_params --objective_2_threshold 100000 --objective_2_type min --max_concurrent 10 --accelerator gpu --data_path /home/sn666/large-scale-data-processing/miniproject/data --use_scheduler --scheduler_max_t 2 --scheduler_strategy nsga_ii --scheduler_grace_period 1 --scheduler_reduction_factor 3 --remark moasha-nsga/maxt2red3/maxaccminparam --use_scaling_config | tee -a "$LOG_FILE"

# # max accuracy and min loss
# LOG_FILE=$(generate_log_file "ax_multiobjective.py")
# echo "Logging to $LOG_FILE"
# CUDA_VISIBLE_DEVICES=2,4,5,6,7 python3 ax_multiobjective.py --num_samples 25 --max_num_epochs 4 --objective_1 ptl/val_accuracy --objective_1_type max --objective_1_threshold 0.90 --objective_2 ptl/val_loss --objective_2_threshold 0.5 --objective_2_type min --max_concurrent 10 --accelerator gpu --data_path /home/sn666/large-scale-data-processing/miniproject/data --use_scheduler --scheduler_max_t 2 --scheduler_strategy nsga_ii --scheduler_grace_period 1 --scheduler_reduction_factor 3 --remark moasha-nsga/maxt2red3/maxaccminloss --use_scaling_config | tee -a "$LOG_FILE"

# # min loss and min params
# LOG_FILE=$(generate_log_file "ax_multiobjective.py")
# echo "Logging to $LOG_FILE"
# CUDA_VISIBLE_DEVICES=2,4,5,6,7 python3 ax_multiobjective.py --num_samples 25 --max_num_epochs 4 --objective_1 ptl/val_loss --objective_1_type min --objective_1_threshold 0.5 --objective_2 ptl/model_params --objective_2_threshold 100000 --objective_2_type min --max_concurrent 10 --accelerator gpu --data_path /home/sn666/large-scale-data-processing/miniproject/data --use_scheduler --scheduler_max_t 2 --scheduler_strategy nsga_ii --scheduler_grace_period 1 --scheduler_reduction_factor 3 --remark moasha-nsga/maxt2red3/minlossminparam --use_scaling_config | tee -a "$LOG_FILE"

# # max accuracy, min loss and min params
# LOG_FILE=$(generate_log_file "ax_multiobjective.py")
# echo "Logging to $LOG_FILE"
# CUDA_VISIBLE_DEVICES=2,4,5,6,7 python3 ax_multiobjective.py --num_samples 25 --max_num_epochs 4 --objective_1 ptl/val_accuracy --objective_1_type max --objective_1_threshold 0.90 --objective_2 ptl/val_loss --objective_2_threshold 0.5 --objective_2_type min --objective_3 ptl/model_params --objective_3_threshold 100000 --objective_3_type min --max_concurrent 10 --accelerator gpu --data_path /home/sn666/large-scale-data-processing/miniproject/data --use_scheduler --scheduler_max_t 2 --scheduler_strategy nsga_ii --scheduler_grace_period 1 --scheduler_reduction_factor 3 --remark moasha-nsga/maxt2red3/maxaccminlossminparam --use_scaling_config | tee -a "$LOG_FILE"

# max accuracy and min params
LOG_FILE=$(generate_log_file "ax_multiobjective.py")
echo "Logging to $LOG_FILE"
CUDA_VISIBLE_DEVICES=2,4,5,6,7 python3 ax_multiobjective.py --num_samples 25 --max_num_epochs 8 --objective_1 ptl/val_accuracy --objective_1_type max --objective_1_threshold 0.90 --objective_2 ptl/model_params --objective_2_threshold 100000 --objective_2_type min --max_concurrent 10 --accelerator gpu --data_path /home/sn666/large-scale-data-processing/miniproject/data --use_scheduler --scheduler_max_t 4 --scheduler_strategy nsga_ii --scheduler_grace_period 1 --scheduler_reduction_factor 3 --remark 8e/moasha-nsga/maxt4red3/maxaccminparam --use_scaling_config | tee -a "$LOG_FILE"

# # max accuracy and min loss
# LOG_FILE=$(generate_log_file "ax_multiobjective.py")
# echo "Logging to $LOG_FILE"
# CUDA_VISIBLE_DEVICES=2,4,5,6,7 python3 ax_multiobjective.py --num_samples 25 --max_num_epochs 8 --objective_1 ptl/val_accuracy --objective_1_type max --objective_1_threshold 0.90 --objective_2 ptl/val_loss --objective_2_threshold 0.5 --objective_2_type min --max_concurrent 10 --accelerator gpu --data_path /home/sn666/large-scale-data-processing/miniproject/data --use_scheduler --scheduler_max_t 4 --scheduler_strategy nsga_ii --scheduler_grace_period 1 --scheduler_reduction_factor 3 --remark 8e/moasha-nsga/maxt4red3/maxaccminloss --use_scaling_config | tee -a "$LOG_FILE"

# # min loss and min params
# LOG_FILE=$(generate_log_file "ax_multiobjective.py")
# echo "Logging to $LOG_FILE"
# CUDA_VISIBLE_DEVICES=2,4,5,6,7 python3 ax_multiobjective.py --num_samples 25 --max_num_epochs 8 --objective_1 ptl/val_loss --objective_1_type min --objective_1_threshold 0.5 --objective_2 ptl/model_params --objective_2_threshold 100000 --objective_2_type min --max_concurrent 10 --accelerator gpu --data_path /home/sn666/large-scale-data-processing/miniproject/data --use_scheduler --scheduler_max_t 4 --scheduler_strategy nsga_ii --scheduler_grace_period 1 --scheduler_reduction_factor 3 --remark 8e/moasha-nsga/maxt4red3/minlossminparam --use_scaling_config | tee -a "$LOG_FILE"

# # max accuracy, min loss and min params
# LOG_FILE=$(generate_log_file "ax_multiobjective.py")
# echo "Logging to $LOG_FILE"
# CUDA_VISIBLE_DEVICES=2,4,5,6,7 python3 ax_multiobjective.py --num_samples 25 --max_num_epochs 8 --objective_1 ptl/val_accuracy --objective_1_type max --objective_1_threshold 0.90 --objective_2 ptl/val_loss --objective_2_threshold 0.5 --objective_2_type min --objective_3 ptl/model_params --objective_3_threshold 100000 --objective_3_type min --max_concurrent 10 --accelerator gpu --data_path /home/sn666/large-scale-data-processing/miniproject/data --use_scheduler --scheduler_max_t 4 --scheduler_strategy nsga_ii --scheduler_grace_period 1 --scheduler_reduction_factor 3 --remark 8e/moasha-nsga/maxt4red3/maxaccminlossminparam --use_scaling_config | tee -a "$LOG_FILE"

# LOG_FILE=$(generate_log_file "ax_multiobjective.py")
# echo "Logging to $LOG_FILE"
# CUDA_VISIBLE_DEVICES=2,4,5,6,7 python3 ax_multiobjective.py --num_samples 25 --max_num_epochs 8 --objective_1 ptl/val_accuracy --objective_1_type max --objective_1_threshold 0.90 --objective_2 ptl/model_params --objective_2_threshold 100000 --objective_2_type min --max_concurrent 10 --accelerator gpu --data_path /home/sn666/large-scale-data-processing/miniproject/data --remark 8e/fifo/maxaccminparam --use_scaling_config | tee -a "$LOG_FILE"

# # max accuracy and min loss
# LOG_FILE=$(generate_log_file "ax_multiobjective.py")
# echo "Logging to $LOG_FILE"
# CUDA_VISIBLE_DEVICES=2,4,5,6,7 python3 ax_multiobjective.py --num_samples 25 --max_num_epochs 8 --objective_1 ptl/val_accuracy --objective_1_type max --objective_1_threshold 0.90 --objective_2 ptl/val_loss --objective_2_threshold 0.5 --objective_2_type min --max_concurrent 10 --accelerator gpu --data_path /home/sn666/large-scale-data-processing/miniproject/data --remark 8e/fifo/maxaccminloss --use_scaling_config | tee -a "$LOG_FILE"

# # min loss and min params
# LOG_FILE=$(generate_log_file "ax_multiobjective.py")
# echo "Logging to $LOG_FILE"
# CUDA_VISIBLE_DEVICES=2,4,5,6,7 python3 ax_multiobjective.py --num_samples 25 --max_num_epochs 8 --objective_1 ptl/val_loss --objective_1_type min --objective_1_threshold 0.5 --objective_2 ptl/model_params --objective_2_threshold 100000 --objective_2_type min --max_concurrent 10 --accelerator gpu --data_path /home/sn666/large-scale-data-processing/miniproject/data --remark 8e/fifo/minlossminparam --use_scaling_config | tee -a "$LOG_FILE"

# # max accuracy, min loss and min params
# LOG_FILE=$(generate_log_file "ax_multiobjective.py")
# echo "Logging to $LOG_FILE"
# CUDA_VISIBLE_DEVICES=2,4,5,6,7 python3 ax_multiobjective.py --num_samples 25 --max_num_epochs 8 --objective_1 ptl/val_accuracy --objective_1_type max --objective_1_threshold 0.90 --objective_2 ptl/val_loss --objective_2_threshold 0.5 --objective_2_type min --objective_3 ptl/model_params --objective_3_threshold 100000 --objective_3_type min --max_concurrent 10 --accelerator gpu --data_path /home/sn666/large-scale-data-processing/miniproject/data --remark 8e/fifo/maxaccminlossminparam --use_scaling_config | tee -a "$LOG_FILE"

PGID=$$

# Trap SIGINT (Ctrl+C) and SIGTERM signals to kill all processes in the group
trap "echo 'Terminating all processes...' | tee -a \"$LOG_FILE\"; kill -TERM -$PGID" SIGINT SIGTERM

# Wait for all background processes to finish
wait

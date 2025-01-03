#!/bin/bash
set -m

# Generate a unique log file for each Python script invocation
generate_log_file() {
  local script_name=$(basename "$1" .py)
  local timestamp=$(date "+%Y%m%d-%H%M%S")
  echo "final_logs/${script_name}_${timestamp}.log"
}

# WITH SCHEDULER
# # max accuracy and min params
LOG_FILE=$(generate_log_file "ax_multiobjective.py")
echo "Logging to $LOG_FILE"
CUDA_VISIBLE_DEVICES=0,1,2,3,6 python3 ax_multiobjective.py --num_samples 25 --max_num_epochs 8 --objective_1 ptl/val_accuracy --objective_1_type max --objective_1_threshold 0.90 --objective_2 ptl/model_params --objective_2_threshold 100000 --objective_2_type min --max_concurrent 10 --accelerator gpu --data_path /home/sn666/large-scale-data-processing/miniproject/data --use_scheduler --scheduler_max_t 8 --scheduler_grace_period 1 --scheduler_reduction_factor 6 --remark 8e/moasha-epsnet/maxt8gr1red6/maxaccminparam --use_scaling_config --results_folder final_results | tee -a "$LOG_FILE"

# # max accuracy and min loss
LOG_FILE=$(generate_log_file "ax_multiobjective.py")
echo "Logging to $LOG_FILE"
CUDA_VISIBLE_DEVICES=0,1,2,3,6 python3 ax_multiobjective.py --num_samples 25 --max_num_epochs 8 --objective_1 ptl/val_accuracy --objective_1_type max --objective_1_threshold 0.90 --objective_2 ptl/val_loss --objective_2_threshold 0.5 --objective_2_type min --max_concurrent 10 --accelerator gpu --data_path /home/sn666/large-scale-data-processing/miniproject/data --use_scheduler --scheduler_max_t 8 --scheduler_grace_period 1 --scheduler_reduction_factor 6 --remark 8e/moasha-epsnet/maxt8gr1red6/maxaccminloss --use_scaling_config --results_folder final_results | tee -a "$LOG_FILE"

# # # min loss and min params
LOG_FILE=$(generate_log_file "ax_multiobjective.py")
echo "Logging to $LOG_FILE"
CUDA_VISIBLE_DEVICES=0,1,2,3,6 python3 ax_multiobjective.py --num_samples 25 --max_num_epochs 8 --objective_1 ptl/val_loss --objective_1_type min --objective_1_threshold 0.5 --objective_2 ptl/model_params --objective_2_threshold 100000 --objective_2_type min --max_concurrent 10 --accelerator gpu --data_path /home/sn666/large-scale-data-processing/miniproject/data --use_scheduler --scheduler_max_t 8 --scheduler_grace_period 1 --scheduler_reduction_factor 6 --remark 8e/moasha-epsnet/maxt8gr1red6/minlossminparam --use_scaling_config --results_folder final_results | tee -a "$LOG_FILE"

# # # max accuracy, min loss and min params
LOG_FILE=$(generate_log_file "ax_multiobjective.py")
echo "Logging to $LOG_FILE"
CUDA_VISIBLE_DEVICES=0,1,2,3,6 python3 ax_multiobjective.py --num_samples 25 --max_num_epochs 8 --objective_1 ptl/val_accuracy --objective_1_type max --objective_1_threshold 0.90 --objective_2 ptl/val_loss --objective_2_threshold 0.5 --objective_2_type min --objective_3 ptl/model_params --objective_3_threshold 100000 --objective_3_type min --max_concurrent 10 --accelerator gpu --data_path /home/sn666/large-scale-data-processing/miniproject/data --use_scheduler --scheduler_max_t 8 --scheduler_grace_period 1 --scheduler_reduction_factor 6 --remark 8e/moasha-epsnet/maxt8gr1red6/maxaccminlossminparam --use_scaling_config --results_folder final_results | tee -a "$LOG_FILE"

# # WITH SCHEDULER
# max accuracy and min params
# LOG_FILE=$(generate_log_file "ax_multiobjective.py")
# echo "Logging to $LOG_FILE"
# CUDA_VISIBLE_DEVICES=0,1,2,3,6 python3 ax_multiobjective.py --num_samples 25 --max_num_epochs 8 --objective_1 ptl/val_accuracy --objective_1_type max --objective_1_threshold 0.90 --objective_2 ptl/model_params --objective_2_threshold 100000 --objective_2_type min --max_concurrent 10 --accelerator gpu --data_path /home/sn666/large-scale-data-processing/miniproject/data --use_scheduler --scheduler_max_t 8 --scheduler_strategy nsga_ii --scheduler_grace_period 1 --scheduler_reduction_factor 6 --remark 8e/moasha-nsga/maxt8gr2red6/maxaccminparam --use_scaling_config --results_folder final_results | tee -a "$LOG_FILE"

# # # max accuracy and min loss
# LOG_FILE=$(generate_log_file "ax_multiobjective.py")
# echo "Logging to $LOG_FILE"
# CUDA_VISIBLE_DEVICES=0,1,2,3,6 python3 ax_multiobjective.py --num_samples 25 --max_num_epochs 8 --objective_1 ptl/val_accuracy --objective_1_type max --objective_1_threshold 0.90 --objective_2 ptl/val_loss --objective_2_threshold 0.5 --objective_2_type min --max_concurrent 10 --accelerator gpu --data_path /home/sn666/large-scale-data-processing/miniproject/data --use_scheduler --scheduler_max_t 8 --scheduler_strategy nsga_ii --scheduler_grace_period 1 --scheduler_reduction_factor 6 --remark 8e/moasha-nsga/maxt8gr2red6/maxaccminloss --use_scaling_config --results_folder final_results | tee -a "$LOG_FILE"

# # # # min loss and min params
# LOG_FILE=$(generate_log_file "ax_multiobjective.py")
# echo "Logging to $LOG_FILE"
# CUDA_VISIBLE_DEVICES=0,1,2,3,6 python3 ax_multiobjective.py --num_samples 25 --max_num_epochs 8 --objective_1 ptl/val_loss --objective_1_type min --objective_1_threshold 0.5 --objective_2 ptl/model_params --objective_2_threshold 100000 --objective_2_type min --max_concurrent 10 --accelerator gpu --data_path /home/sn666/large-scale-data-processing/miniproject/data --use_scheduler --scheduler_max_t 8 --scheduler_strategy nsga_ii --scheduler_grace_period 1 --scheduler_reduction_factor 6 --remark 8e/moasha-nsga/maxt8gr2red6/minlossminparam --use_scaling_config --results_folder final_results | tee -a "$LOG_FILE"

# # # # max accuracy, min loss and min params
# LOG_FILE=$(generate_log_file "ax_multiobjective.py")
# echo "Logging to $LOG_FILE"
# CUDA_VISIBLE_DEVICES=0,1,2,3,6 python3 ax_multiobjective.py --num_samples 25 --max_num_epochs 8 --objective_1 ptl/val_accuracy --objective_1_type max --objective_1_threshold 0.90 --objective_2 ptl/val_loss --objective_2_threshold 0.5 --objective_2_type min --objective_3 ptl/model_params --objective_3_threshold 100000 --objective_3_type min --max_concurrent 10 --accelerator gpu --data_path /home/sn666/large-scale-data-processing/miniproject/data --use_scheduler --scheduler_max_t 8 --scheduler_strategy nsga_ii --scheduler_grace_period 1 --scheduler_reduction_factor 6 --remark 8e/moasha-nsga/maxt8gr2red6/maxaccminlossminparam --use_scaling_config --results_folder final_results | tee -a "$LOG_FILE"

# WITHOUT SCHEDULER
# max accuracy and min params
# LOG_FILE=$(generate_log_file "ax_multiobjective.py")
# echo "Logging to $LOG_FILE"
# CUDA_VISIBLE_DEVICES=0,1,2,3,6 python3 ax_multiobjective.py --num_samples 25 --max_num_epochs 8 --objective_1 ptl/val_accuracy --objective_1_type max --objective_1_threshold 0.90 --objective_2 ptl/model_params --objective_2_threshold 100000 --objective_2_type min --max_concurrent 10 --accelerator gpu --data_path /home/sn666/large-scale-data-processing/miniproject/data --remark 8e/fifo/maxaccminparam --use_scaling_config --results_folder final_results | tee -a "$LOG_FILE"

# # # max accuracy and min loss
# LOG_FILE=$(generate_log_file "ax_multiobjective.py")
# echo "Logging to $LOG_FILE"
# CUDA_VISIBLE_DEVICES=0,1,2,3,6 python3 ax_multiobjective.py --num_samples 25 --max_num_epochs 8 --objective_1 ptl/val_accuracy --objective_1_type max --objective_1_threshold 0.90 --objective_2 ptl/val_loss --objective_2_threshold 0.5 --objective_2_type min --max_concurrent 10 --accelerator gpu --data_path /home/sn666/large-scale-data-processing/miniproject/data --remark 8e/fifo/maxaccminloss --use_scaling_config --results_folder final_results | tee -a "$LOG_FILE"

# # # min loss and min params
# LOG_FILE=$(generate_log_file "ax_multiobjective.py")
# echo "Logging to $LOG_FILE"
# CUDA_VISIBLE_DEVICES=0,1,2,3,6 python3 ax_multiobjective.py --num_samples 25 --max_num_epochs 8 --objective_1 ptl/val_loss --objective_1_type min --objective_1_threshold 0.5 --objective_2 ptl/model_params --objective_2_threshold 100000 --objective_2_type min --max_concurrent 10 --accelerator gpu --data_path /home/sn666/large-scale-data-processing/miniproject/data --remark 8e/fifo/minlossminparam --use_scaling_config --results_folder final_results | tee -a "$LOG_FILE"

# # # max accuracy, min loss and min params
# LOG_FILE=$(generate_log_file "ax_multiobjective.py")
# echo "Logging to $LOG_FILE"
# CUDA_VISIBLE_DEVICES=0,1,2,3,6 python3 ax_multiobjective.py --num_samples 25 --max_num_epochs 8 --objective_1 ptl/val_accuracy --objective_1_type max --objective_1_threshold 0.90 --objective_2 ptl/val_loss --objective_2_threshold 0.5 --objective_2_type min --objective_3 ptl/model_params --objective_3_threshold 100000 --objective_3_type min --max_concurrent 10 --accelerator gpu --data_path /home/sn666/large-scale-data-processing/miniproject/data --remark 8e/fifo/maxaccminlossminparam --use_scaling_config --results_folder final_results | tee -a "$LOG_FILE"

PGID=$$

# Trap SIGINT (Ctrl+C) and SIGTERM signals to kill all processes in the group
trap "echo 'Terminating all processes...' | tee -a \"$LOG_FILE\"; kill -TERM -$PGID" SIGINT SIGTERM

# Wait for all background processes to finish
wait

#!/bin/sh
export BABYAI_STORAGE='storage'
eval "$(conda shell.bash hook)"
conda activate ella
python babyai/scripts/train_subtask_prediction_model.py \
--demos $1_agent_subtasks_oracle \
--arch siamese-l1 \
--episodes 100 \
--val-episodes 50 \
--low-level-demos $3_agent_done \
--ll-episodes 150 \
--log-interval 1 \
--save-interval 10 \
--val-interval 5 \
--batch-size 10 \
--denoise --denoise-k 1 \
--epochs 20 \
--ones \
--model $2-Subtask-Classifier-Ones \
#--wb

#!/bin/sh
export BABYAI_STORAGE='storage'
eval "$(conda shell.bash hook)"
conda activate ella
python babyai/scripts/train_rl.py \
--arch expert_filmcnn \
--env $1 \
--demos $3_agent_done \
--episodes 15000 \
--reward-shaping subtask_classifier_online --subtask-discount 0.99 \
--subtask-arch siamese-l1 \
--subtask-pretrained-model $2-Subtask-Classifier-Ones \
--subtask-batch-size 10 --subtask-update-rate 50 --subtask-updates 3 \
--subtask-hl-demos $1_agent_subtasks_oracle --subtask-val-episodes 100 \
--pi-l-scale 0.25 --reward-scale 20 \
--hrl shape \
--pi-l $4-Done-Classifier_best --done-classifier \
--log-interval 1 --save-interval 15 --val-interval 15 \
--val-episodes 128 --procs 64 --frames-per-proc 40 --recurrence 20 \
--seed $5 \
--model onl/$2-$4-RS-Online-025-D-$5 \
#--wb

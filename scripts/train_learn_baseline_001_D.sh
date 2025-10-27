#!/bin/sh
export BABYAI_STORAGE='storage'
eval "$(conda shell.bash hook)"
conda activate ella
python babyai/scripts/train_rl.py \
--arch expert_filmcnn \
--env $1 \
--demos $3_agent_done \
--episodes 15000 \
--reward-shaping learn_baseline \
--pi-l-scale 0.01 --reward-scale 20 \
--hrl shape \
--pi-l GTL-Done-Classifier_best --done-classifier \
--log-interval 1 --save-interval 15 --val-interval 15 \
--val-episodes 128 --procs 64 --frames-per-proc 40 --recurrence 20 \
--learn-baseline $4-LEARN-Classifier \
--seed $5 \
--subtask-discount 0.99 \
--model $2-RS-LEARN-Baseline-001-D-$5 \
#--wb

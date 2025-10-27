#!/bin/sh
export BABYAI_STORAGE='storage'
# eval "$(conda shell.bash hook)"
# conda activate ella
python babyai/scripts/train_rl.py \
--arch expert_filmcnn \
--env $1 \
--episodes 15000 \
--reward-shaping RIDE --subtask-discount 0.99 \
--pi-l-scale $6 --reward-scale $4 \
--hrl shape \
--log-interval 1 --save-interval 15 --val-interval 15 \
--val-episodes 128 --procs ${8} --frames-per-proc 40 --recurrence 20 \
--seed $3 \
--model QG_QA/$2-RIDE-reward_scale_$5-lambda_$7-seed_$3 \

#--wb

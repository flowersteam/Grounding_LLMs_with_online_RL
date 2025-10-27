#!/bin/sh
export BABYAI_STORAGE='storage'
# eval "$(conda shell.bash hook)"
# conda activate ella
python babyai/scripts/train_rl.py \
--arch expert_filmcnn \
--env $1 \
--train-env $5 \
--episodes 15000 \
--reward-shaping QG_QA --subtask-discount 0.99 \
--pi-l-scale $9 --reward-scale 20 \
--hrl shape \
--log-interval 1 --save-interval 15 --val-interval 15 \
--val-episodes 128 --procs ${11} --frames-per-proc 40 --recurrence 20 \
--seed $4 \
--model QG_QA/$2-$3-train_env-$6_no_answer-lambda_${10}-model-$7_$8-seed_$4 \
--type-QG-QA-reward $3 \
--model-QA $7 \
--epoch-QA $8 \
--no-answer-question True

#--wb

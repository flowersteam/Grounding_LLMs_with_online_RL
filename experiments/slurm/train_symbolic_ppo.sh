#!/bin/sh
export BABYAI_STORAGE='storage'
export DLP_STORAGE='storage'
python -m experiments.train_symbolic_ppo \
--arch expert_filmcnn \
--env $1 \
--hrl vanilla \
--log-interval 1 --save-interval 15  --val-interval 15 --val-episodes 128 \
--procs 64 --frames-per-proc 40 --recurrence 20 \
--seed $4 \
--number-actions $3 \
--frames 400000 \
--model $2-nbr_actions-$3-PPO-NoPre-$4 \
#--wb

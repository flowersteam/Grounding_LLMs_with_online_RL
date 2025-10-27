#!/bin/sh
export BABYAI_STORAGE='storage'
eval "$(conda shell.bash hook)"
conda activate ella
python babyai/scripts/train_learn_baseline_model.py \
--demos $1_agent \
--episodes 15000 --val-episodes=200 \
--log-interval 1 --save-interval 5 --val-interval 5 \
--epochs 100 \
--model $2-LEARN-Classifier \
#--wb

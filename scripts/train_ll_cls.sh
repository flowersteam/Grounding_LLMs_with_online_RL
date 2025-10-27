#!/bin/sh
export BABYAI_STORAGE='storage'
eval "$(conda shell.bash hook)"
conda activate ella
python babyai/scripts/train_il.py \
--env $1 \
--demos $1_agent_done \
--episodes 15000 --val-episodes 200 \
--batch-size 2560 \
--log-interval 1 --save-interval 5 --val-interval 1 \
--include-done --done-classifier --oversample 70 \
--epochs 5 \
--model $2-Done-Classifier \
#--wb

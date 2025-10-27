#!/bin/sh
export BABYAI_STORAGE='storage'
eval "$(conda shell.bash hook)"
conda activate ella
python babyai/scripts/train_l_class.py \
--env $1 \
--demos $1_agent_done \
--episodes 15000 --val-episodes 200 \
--batch-size $3 \
--log-interval 1 --save-interval 5 --val-interval 1 \
--include-done --QA --oversample 70 \
--epochs 40 \
--model $2-QA \
--model-number 0 \
--lr 1e-4 \
#--wb
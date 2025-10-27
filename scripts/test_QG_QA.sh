#!/bin/sh
export BABYAI_STORAGE='storage'
eval "$(conda shell.bash hook)"
conda activate ella
python babyai/scripts/test_QG_QA.py \
--env $1 \
--episodes 1500 --valid-episodes 150 \
--include-goal --QG-generation \
--seed 1 --name-env-short $2 --test-QA-env
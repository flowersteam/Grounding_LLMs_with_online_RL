#!/bin/sh
export BABYAI_STORAGE='storage'
eval "$(conda shell.bash hook)"
conda activate ella
python babyai/scripts/make_agent_demos.py \
--env $1 \
--episodes 15000 --valid-episodes 1000 \
--include-goal \
--seed 1

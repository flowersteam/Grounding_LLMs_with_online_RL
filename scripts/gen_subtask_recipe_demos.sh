#!/bin/sh
export BABYAI_STORAGE='storage'
eval "$(conda shell.bash hook)"
conda activate ella
python babyai/scripts/make_subtask_recipe_demos.py \
--env $1 \
--low-level-demos $3_agent_done \
--ll-episodes 15000 \
--episodes 500 --valid-episodes 500 \
--oracle --nonstrict

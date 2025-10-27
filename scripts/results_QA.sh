#!/bin/sh
export BABYAI_STORAGE='storage'
eval "$(conda shell.bash hook)"
conda activate ella
python babyai/scripts/result_QA_study.py
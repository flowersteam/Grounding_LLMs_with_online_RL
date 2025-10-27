#!/bin/sh
export BABYAI_STORAGE='storage'

echo $SLURM_STEP_GPUS
python3 -u babyai/babyai/test_paral.py
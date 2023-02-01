#!/bin/bash

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | sed -n "1p")
echo "running process ${SLURM_PROCID} on node $(hostname) with master ${MASTER_ADDR}"
export "DLP_STORAGE"='storage'
python -m lamorel_launcher.launch lamorel_args.accelerate_args.machine_rank=$SLURM_PROCID lamorel_args.accelerate_args.main_process_ip=$MASTER_ADDR $*
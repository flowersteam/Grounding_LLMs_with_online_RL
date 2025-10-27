#!/bin/bash

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | sed -n "1p")
echo "running on node $(hostname)"
accelerate launch --machine_rank $SLURM_PROCID --main_process_ip $MASTER_ADDR --main_process_port 13370 $*
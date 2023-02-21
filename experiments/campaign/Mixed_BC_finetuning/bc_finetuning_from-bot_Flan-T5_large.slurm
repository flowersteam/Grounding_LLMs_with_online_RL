#!/bin/bash
#SBATCH --job-name=bc_finetuning_from-bot_Flan-T5_large_seed_%a  # job name
#SBATCH --time=04:00:00  # maximum execution time (HH:MM:SS)
#SBATCH --output=slurm_logs/bc_finetuning_from-bot_Flan-T5_large_seed_%a-%j.out # output
#SBATCH --error=slurm_logs/bc_finetuning_from-bot_Flan-T5_large_seed_%a-%j.err # err
#SBATCH --account= # SLURM ACCOUNT
#SBATCH --qos=qos_gpu-t3
#SBATCH -C a100
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --hint=nomultithread
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1

#SBATCH --array=1-2

module purge
module load python/3.8.2
conda activate dlp

chmod +x experiments/slurm/accelerate_launcher.sh

srun experiments/slurm/accelerate_launcher.sh \
    --config_file $WORK/Grounding_LLMs/experiments/configs/accelerate/default_config.yaml \
    --multi_gpu \
    --num_processes 8 \
    --num_machines 1 \
    experiments/clm_behavioral-cloning.py \
    --output_dir=$WORK/Grounding_LLMs/storage/logs/bc_finetuning_Flan-T5_large_seed_${SLURM_ARRAY_TASK_ID} \
    --model_dir=$SCRATCH/Grounding_LLMs/llms/flan-t5-large \
    --data_dir=$WORK/Grounding_LLMs/storage/logs/bot_GoToLocal_seed_${SLURM_ARRAY_TASK_ID}/test/BabyAI-GoToLocal-v0/return_per_episode \
    --per_device_batch_size=8 \
    --gradient_accumulation_steps=1 \
    --seed=${SLURM_ARRAY_TASK_ID} \
    --file_name=bot_trajectories \
    --file_id=1
                    

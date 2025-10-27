#!/bin/bash
#SBATCH --job-name=clm_finetuning_from-bot_Flan_T5large_%a  # job name
#SBATCH --time=04:00:00  # maximum execution time (HH:MM:SS)
#SBATCH --output=slurm_logs/clm_finetuning_from-bot_Flan_T5large_%a-%j.out # output
#SBATCH --error=slurm_logs/clm_finetuning_from-bot_Flan_T5large_%a-%j.err # err
#SBATCH --account=imi@a100
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

chmod +x dlp/slurm/accelerate_launcher.sh

srun dlp/slurm/accelerate_launcher.sh \
    --config_file $WORK/DLP/dlp/configs/accelerate/default_config.yaml \
    --multi_gpu \
    --num_processes 8 \
    --num_machines 1 \
    dlp/clm_finetuning.py \
    --output_dir=$WORK/DLP/storage/logs/clm_finetuning_from-bot_FLan_T5large_seed_${SLURM_ARRAY_TASK_ID} \
    --model_dir=/gpfsscratch/rech/imi/ucy39hi/saycan-scienceworld/llms/flan-t5-large \
    --data_dir=/gpfswork/rech/imi/uez56by/code/DLP/storage/logs/llm_gtl_nbr_env_2_Flan_T5large_pretrained_True_nbr_actions_6_turn_left_turn_right_go_forward_pick_up_drop_toggle_shape_reward_beta_0_seed_${SLURM_ARRAY_TASK_ID}/test/BabyAI-GoToLocal-v0/return_per_episode \
    --per_device_batch_size=8 \
    --gradient_accumulation_steps=1 \
    --seed=${SLURM_ARRAY_TASK_ID} \
    --file_name=bot_trajectories \
    --file_id=1
                    

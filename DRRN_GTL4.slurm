#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --account=imi@v100
#SBATCH --job-name=DRRN_GTL4_%a
#SBATCH -o slurm_logs/DRRN_GTL4_%a.out
#SBATCH -e slurm_logs/DRRN_GTL4_%a.err
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --array=1-2
#SBATCH --qos=qos_gpu-t3
#SBATCH -C v100-32g

module purge
module load python/3.8.2
conda activate dlp

srun python dlp/main.py \
		    rl_script_args.seed=${SLURM_ARRAY_TASK_ID} \
                    rl_script_args.number_envs=32 \
                    rl_script_args.num_steps=1000000 \
                    rl_script_args.action_space=["turn_left","turn_right","go_forward","pick_up","drop","toggle"] \
                    rl_script_args.saving_path_logs=$WORK/code/DLP/storage/logs \
                    rl_script_args.name_experiment='drrn_gtl_distractor_4' \
                    rl_script_args.name_model='DRRN' \
                    rl_script_args.name_environment='BabyAI-GoToLocalS8N4-v0' \
                    rl_script_args.saving_path_model=$SCRATCH/DLP/models \
                    rl_script_args.spm_path=$WORK/code/DLP/dlp/agents/drrn/spm_models/unigram_8k.model \
                    lamorel_args.distributed_setup_args.n_llm_processes=0 \
                    --config-path=$WORK/code/DLP/dlp/configs \
                    --config-name=multi-node_slurm_cluster_config 

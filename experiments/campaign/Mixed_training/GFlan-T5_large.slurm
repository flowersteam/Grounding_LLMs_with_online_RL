#!/bin/bash
#SBATCH --job-name=GFlan-T5_large_seed_%a    # job name
#SBATCH --time=20:00:00 # maximum execution time (HH:MM:SS)
#SBATCH --output=slurm_logs/GFlan-T5_large_seed_%a-%j.out     # output file name
#SBATCH --error=slurm_logs/GFlan-T5_large_seed_%a-%j.err      # err file name
#SBATCH --account= # SLURM ACCOUNT
#SBATCH --qos=qos_gpu-t3
#SBATCH -C a100
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --hint=nomultithread
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1

#SBATCH --array=1-2

module purge
module load python/3.8.2
conda activate dlp

chmod +x experiments/slurm/launcher.sh

srun experiments/slurm/launcher.sh \
                    rl_script_args.path=$WORK/Grounding_LLMs/experiments/train_language_agent.py \
                    rl_script_args.seed=${SLURM_ARRAY_TASK_ID} \
                    rl_script_args.number_envs=32 \
                    rl_script_args.num_steps=1500000 \
                    rl_script_args.action_space=["turn_left","turn_right","go_forward","pick_up","drop","toggle"] \
                    rl_script_args.saving_path_logs=$WORK/Grounding_LLMs/storage/logs \
                    rl_script_args.name_experiment='llm_mtrl' \
                    rl_script_args.name_model='Flan_T5large' \
                    rl_script_args.name_environment='BabyAI-MixedTrainLocal-v0' \
                    rl_script_args.template_test=1 \
                    rl_script_args.saving_path_model=$SCRATCH/Grounding_LLMs/models \
                    lamorel_args.llm_args.model_type=seq2seq \
                    lamorel_args.llm_args.model_path=$SCRATCH/Grounding_LLMs/llms/flan-t5-large \
                    lamorel_args.llm_args.parallelism.model_parallelism_size=2 \
                    lamorel_args.llm_args.minibatch_size=3 \
                    lamorel_args.accelerate_args.num_machines=1 \
                    --config-path=$WORK/Grounding_LLMs/experiments/configs \
                    --config-name=multi-node_slurm_cluster_config 

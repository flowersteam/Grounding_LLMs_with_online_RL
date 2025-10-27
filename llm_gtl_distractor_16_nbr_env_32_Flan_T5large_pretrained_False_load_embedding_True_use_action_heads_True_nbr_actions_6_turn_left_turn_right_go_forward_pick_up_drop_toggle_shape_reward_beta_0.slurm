#!/bin/bash
#SBATCH --job-name=llm_gtl_distractor_16_nbr_env_32_Flan_T5large_pretrained_False_load_embedding_True_use_action_heads_True_nbr_actions_6_turn_left_turn_right_go_forward_pick_up_drop_toggle_shape_reward_beta_0_seed_%a # job name
#SBATCH --time=20:00:00             										        					     # maximum execution time (HH:MM:SS)
#SBATCH --output=slurm_logs/llm_gtl_distractor_16_nbr_env_32_Flan_T5large_pretrained_False_load_embedding_True_use_action_heads_True_nbr_actions_6_turn_left_turn_right_go_forward_pick_up_drop_toggle_shape_reward_beta_0_seed_%a-%j.out     # output file name
#SBATCH --error=slurm_logs/llm_gtl_distractor_16_nbr_env_32_Flan_T5large_pretrained_False_load_embedding_True_use_action_heads_True_nbr_actions_6_turn_left_turn_right_go_forward_pick_up_drop_toggle_shape_reward_beta_0_seed_%a-%j.err      # err file name
#SBATCH --account=imi@a100
#SBATCH --qos=qos_gpu-t3
#SBATCH -C a100
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1

#SBATCH --array=1-2

module purge
module load python/3.8.2
conda activate dlp

chmod +x dlp/slurm/launcher.sh

srun dlp/slurm/launcher.sh \
                    rl_script_args.path=$WORK/code/DLP/dlp/main.py \
                    rl_script_args.seed=${SLURM_ARRAY_TASK_ID} \
                    rl_script_args.number_envs=32 \
                    rl_script_args.num_steps=400000 \
                    rl_script_args.action_space=["turn_left","turn_right","go_forward","pick_up","drop","toggle"] \
                    rl_script_args.saving_path_logs=$WORK/code/DLP/storage/logs \
                    rl_script_args.name_experiment='llm_gtl_distractor_16' \
                    rl_script_args.name_model='Flan_T5large' \
                    rl_script_args.name_environment='BabyAI-GoToLocalS8N16-v0' \
                    rl_script_args.template_test=2 \
                    rl_script_args.saving_path_model=$SCRATCH/DLP/models \
                    rl_script_args.load_embedding=true \
                    rl_script_args.use_action_heads=true \
                    lamorel_args.llm_args.model_type=seq2seq \
                    lamorel_args.llm_args.model_path=/gpfsscratch/rech/imi/ucy39hi/saycan-scienceworld/llms/flan-t5-large \
                    lamorel_args.llm_args.pretrained=false \
                    lamorel_args.llm_args.model_parallelism_size=2 \
                    lamorel_args.llm_args.minibatch_size=3 \
                    lamorel_args.accelerate_args.num_machines=1 \
                    lamorel_args.accelerate_args.num_processes=5 \
                    lamorel_args.allow_subgraph_use_whith_gradient=false \
                    --config-path=$WORK/code/DLP/dlp/configs \
                    --config-name=multi-node_slurm_cluster_config 

#!/bin/bash
#SBATCH --job-name=test_llm_gtl_nbr_env_32_Flan_T5large_pretrained_True_nbr_actions_6_turn_left_turn_right_go_forward_pick_up_drop_toggle_shape_reward_beta_0_seed_%a 		       # job name
#SBATCH --time=07:30:00             										        					      # maximum execution time (HH:MM:SS)
#SBATCH --output=slurm_logs/test_llm_gtl_nbr_env_32_Flan_T5large_pretrained_True_nbr_actions_6_turn_left_turn_right_go_forward_pick_up_drop_toggle_shape_reward_beta_0_%a-%j.out      # output file name
#SBATCH --error=slurm_logs/test_llm_gtl_nbr_env_32_Flan_T5large_pretrained_True_nbr_actions_6_turn_left_turn_right_go_forward_pick_up_drop_toggle_shape_reward_beta_0_%a-%j.err       # err file name
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
                    rl_script_args.path=$WORK/code/DLP/dlp/main_test.py \
                    rl_script_args.seed=${SLURM_ARRAY_TASK_ID} \
                    rl_script_args.number_envs=32 \
                    rl_script_args.number_episodes=1000 \
                    rl_script_args.action_space=["turn_left","turn_right","go_forward","pick_up","drop","toggle"] \
                    rl_script_args.saving_path_logs=$WORK/code/DLP/storage/logs \
                    rl_script_args.name_experiment='llm_gtl' \
                    rl_script_args.name_model='Flan_T5large' \
                    rl_script_args.zero_shot=False \
                    rl_script_args.saving_path_model=$SCRATCH/DLP/models \
                    lamorel_args.llm_args.model_type=seq2seq \
                    lamorel_args.llm_args.model_path=/gpfsscratch/rech/imi/ucy39hi/saycan-scienceworld/llms/flan-t5-large \
                    lamorel_args.llm_args.model_parallelism_size=2 \
                    lamorel_args.llm_args.minibatch_size=3 \
                    lamorel_args.accelerate_args.num_machines=1 \
                    lamorel_args.accelerate_args.num_processes=5 \
                    --config-path=$WORK/code/DLP/dlp/configs \
                    --config-name=multi-node_slurm_cluster_config_test
                    

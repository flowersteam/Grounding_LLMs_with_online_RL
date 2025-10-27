#!/bin/bash
#SBATCH --job-name=test-lamorel
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=2
#SBATCH --cpus-per-task=20
#SBATCH --hint=nomultithread
#SBATCH --time=00:05:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
#SBATCH --account= # Set this
#SBATCH --qos= # Set this
#SBATCH -C v100-16g
#SBATCH --gres=gpu:4

chmod +x examples/slurm/launcher.sh
cd examples

srun slurm/launcher.sh \
    rl_script_args.path=YOUR_PATH/examples/example_script.py \
    --config-path=YOUR_PATH/examples/configs \
    --config-name=slurm_cluster_single-node_config
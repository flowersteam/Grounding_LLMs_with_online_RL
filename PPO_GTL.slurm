#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --account=imi@v100
#SBATCH --job-name=PPO_GTL_%a
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH -o slurm_logs/PPO_GTL_%a.out
#SBATCH -e slurm_logs/PPO_GTL_%a.err
#SBATCH --array=1-2
#SBATCH --qos=qos_gpu-t3
#SBATCH -C v100-32g

module purge
module load python/3.8.2
conda activate dlp

srun scripts/train_ppo_baseline.sh BabyAI-GoToLocal-v0 GTL 6 ${SLURM_ARRAY_TASK_ID}

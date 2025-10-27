#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --account=imi@v100
#SBATCH --job-name=PPO_GTRB_%a
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH -o slurm_logs/PPO_GTRB_%a.out
#SBATCH -e slurm_logs/PPO_GTRB_%a.err
#SBATCH --array=1-2
#SBATCH --qos=qos_gpu-t3
#SBATCH -C v100-32g

module purge
module load python/3.8.2
conda activate dlp

srun scripts/train_ppo_baseline.sh BabyAI-GoToRedBallNoDists-v0 GTRB 6 ${SLURM_ARRAY_TASK_ID}

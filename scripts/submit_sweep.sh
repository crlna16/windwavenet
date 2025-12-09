#!/bin/bash
#SBATCH -A ka1176
#SBATCH -p vader
#SBATCH --array=0-1
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH -o scripts/slurm/slurm-%j.out

module load git
uv sync
source .venv/bin/activate
#wandb agent wwm-emulator/simple-wwm-emulator/w5i9gsye
#wandb agent wwm-emulator/simple-wwm-emulator/pneifth9
#wandb agent wwm-emulator/simple-wwm-emulator/na1becz6
#wandb agent wwm-emulator/simple-wwm-emulator/ju8o025p
wandb agent wwm-emulator/simple-wwm-emulator/ql1ymdn4

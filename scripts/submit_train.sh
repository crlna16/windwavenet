#!/bin/bash
#SBATCH -A ka1176
#SBATCH -p vader
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH -o scripts/slurm/slurm-%j.out

uv run src/train.py experiment=best_sweep_ju8o025p
#uv run src/train.py
#uv run src/train.py model.backbone.input_dim=2 data.features='["windSpeedX", "windSpeedY"]'

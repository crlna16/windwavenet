#!/bin/bash
#SBATCH -A ka1176
#SBATCH -p vader
#SBATCH --mem=500G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH -o scripts/slurm/slurm-%j.out

uv run src/train.py experiment=densenet_1h data.train_days="[1, 120]"
#uv run src/train.py
#uv run src/train.py model.backbone.input_dim=2 data.features='["windSpeedX", "windSpeedY"]'

#!/bin/bash
#SBATCH -A ka1176
#SBATCH -p gpu
#SBATCH --exclusive
#SBATCH --time=08:00:00
#SBATCH -o scripts/slurm/slurm-%j.out

uv run src/train.py experiment=wwnet2_lstm trainer.devices="auto"

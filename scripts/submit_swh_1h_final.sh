#!/bin/bash
#SBATCH -A ka1176
#SBATCH -p gpu
#SBATCH --mem=200G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH -o scripts/slurm/slurm-%j.out

# LSTM
uv run python scripts/train.py experiment=swh_1h_lstm data.seq_len=48
uv run python scripts/train.py experiment=swh_1h_lstm data.seq_len=24
uv run python scripts/train.py experiment=swh_1h_lstm data.seq_len=12
uv run python scripts/train.py experiment=swh_1h_lstm data.seq_len=6
uv run python scripts/train.py experiment=swh_1h_lstm data.seq_len=3
uv run python scripts/train.py experiment=swh_1h_lstm data.seq_len=2

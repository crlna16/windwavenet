#!/bin/bash
#SBATCH -A ka1176
#SBATCH -p gpu
#SBATCH --mem=0
#SBATCH --gres=gpu:2
#SBATCH --exclusive
#SBATCH --time=08:00:00
#SBATCH -o scripts/slurm/slurm-%j.out

# DenseNet
#uv run python src/train.py experiment=swh_1h_densenet

# LSTM
# DONE uv run python src/train.py experiment=swh_1h_lstm data.seq_len=48
# DONE uv run python src/train.py experiment=swh_1h_lstm data.seq_len=24
# DONE uv run python src/train.py experiment=swh_1h_lstm data.seq_len=12
# DONE uv run python src/train.py experiment=swh_1h_lstm data.seq_len=6
# DONE uv run python src/train.py experiment=swh_1h_lstm data.seq_len=3
# DONE uv run python src/train.py experiment=swh_1h_lstm data.seq_len=2

# Multi-Head GAT
uv run python src/train.py experiment=swh_1h_multi-head-gat.yaml trainer.devices=2 data.max_nodes=5
uv run python src/train.py experiment=swh_1h_multi-head-gat.yaml trainer.devices=2 data.max_nodes=50
uv run python src/train.py experiment=swh_1h_multi-head-gat.yaml trainer.devices=2 data.max_nodes=500
uv run python src/train.py experiment=swh_1h_multi-head-gat.yaml trainer.devices=2 data.max_nodes=5000 data.batch_size=26

#!/bin/bash
#SBATCH -A ka1176
#SBATCH -p gpu
#SBATCH --mem=0
#SBATCH --gres=gpu:2
#SBATCH --exclusive
#SBATCH --time=08:00:00
#SBATCH -o scripts/slurm/slurm-%j.out

# DenseNet
# DONE uv run python src/train.py experiment=swh_1h_densenet trainer.devices=2

# LSTM
# DONE uv run python src/train.py experiment=swh_1h_lstm data.seq_len=48
# DONE uv run python src/train.py experiment=swh_1h_lstm data.seq_len=24
# DONE uv run python src/train.py experiment=swh_1h_lstm data.seq_len=12
# DONE uv run python src/train.py experiment=swh_1h_lstm data.seq_len=6
# DONE uv run python src/train.py experiment=swh_1h_lstm data.seq_len=3
# DONE uv run python src/train.py experiment=swh_1h_lstm data.seq_len=2
# DONE uv run python src/train.py experiment=swh_1h_lstm data.seq_len=16 trainer.devices=2
# DONE uv run python src/train.py experiment=swh_1h_lstm data.seq_len=20 trainer.devices=2
# DONE uv run python src/train.py experiment=swh_1h_lstm data.seq_len=28 trainer.devices=2

# Multi-Head GAT
# DONE uv run python src/train.py experiment=swh_1h_multi-head-gat.yaml trainer.devices=2 data.max_nodes=5
# DONE uv run python src/train.py experiment=swh_1h_multi-head-gat.yaml trainer.devices=2 data.max_nodes=50 test=false predict=false
# DONE uv run python src/train.py experiment=swh_1h_multi-head-gat.yaml trainer.devices=2 data.max_nodes=500 test=false predict=false
# DONE uv run python src/train.py experiment=swh_1h_multi-head-gat.yaml trainer.devices=2 data.max_nodes=1000 test=false predict=false
# DONE uv run python src/train.py experiment=swh_1h_multi-head-gat.yaml trainer.devices=2 data.max_nodes=5000 data.batch_size=26 +trainer.num_nodes=1 test=false predict=false

# Multi-Head GAT VN
# RUNNING uv run python src/train.py experiment=swh_1h_multi-head-gat-vn.yaml trainer.devices=2 test=false
# RUNNING uv run python src/train.py experiment=swh_1h_multi-head-gat-vn.yaml trainer.devices=2 test=false model.backbone.dropout=0.1 model.backbone.attn_dropout=0.2

# Multi-Head GAT LSTM VN
uv run python src/train.py experiment=swh_1h_multi-head-gat-lstm-vn.yaml trainer.devices=2 test=false

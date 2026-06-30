#!/bin/bash
#SBATCH -A ka1176
#SBATCH -p gpu
#SBATCH --mem=0
#SBATCH --gres=gpu:3
#SBATCH --exclusive
#SBATCH --time=08:00:00
#SBATCH -o scripts/slurm/slurm-%A_%a.out
#SBATCH --array=0-34

# how to call
# sbatch --array=15-18 scripts/submit_swh_mwp_mwd_1h_array.sh

COMMANDS=(
    # DenseNet (0)
    "uv run python src/train.py experiment=swh_mwp_mwd_1h_densenet trainer.devices=3"
    # LSTM (1-6)
    # Discarded "uv run python src/train.py experiment=swh_mwp_mwd_1h_lstm data.seq_len=2  trainer.devices=3"
    "uv run python src/train.py experiment=swh_mwp_mwd_1h_lstm data.seq_len=3  trainer.devices=3"
    "uv run python src/train.py experiment=swh_mwp_mwd_1h_lstm data.seq_len=6  trainer.devices=3"
    "uv run python src/train.py experiment=swh_mwp_mwd_1h_lstm data.seq_len=12 trainer.devices=3"
    "uv run python src/train.py experiment=swh_mwp_mwd_1h_lstm data.seq_len=24 trainer.devices=3"
    "uv run python src/train.py experiment=swh_mwp_mwd_1h_lstm data.seq_len=48 trainer.devices=3"
    # Multi-Head GAT (7-9)
    # Discarded "uv run python src/train.py experiment=swh_mwp_mwd_1h_multi-head-gat.yaml trainer.devices=3 data.max_nodes=250 test=false predict=false"
    # Discarded "uv run python src/train.py experiment=swh_mwp_mwd_1h_multi-head-gat.yaml trainer.devices=3 data.max_nodes=500 test=false predict=false"
    # Discarded "uv run python src/train.py experiment=swh_mwp_mwd_1h_multi-head-gat.yaml trainer.devices=3 data.max_nodes=1000 test=false predict=false"
    # Multi-Head GAT VN (10-14)
    "uv run python src/train.py experiment=swh_mwp_mwd_1h_multi-head-gat-vn.yaml trainer.devices=3 test=false"
    "uv run python src/train.py experiment=swh_mwp_mwd_1h_multi-head-gat-vn.yaml trainer.devices=3 test=false data.max_nodes=50"
    "uv run python src/train.py experiment=swh_mwp_mwd_1h_multi-head-gat-vn.yaml trainer.devices=3 test=false data.max_nodes=250"
    # Multi-Head GAT LSTM VN (15-17)
    # Excluded "uv run python src/train.py experiment=swh_mwp_mwd_1h_multi-head-gat-lstm-vn.yaml trainer.devices=3 test=false"
    # Discarded "uv run python src/train.py experiment=swh_mwp_mwd_1h_multi-head-gat-lstm-vn.yaml trainer.devices=3 test=false model.backbone.dropout=0.1 model.backbone.heads=4 model.backbone.hidden_dim=256 model.backbone.num_layers=8"
    # Discarded "uv run python src/train.py experiment=swh_mwp_mwd_1h_multi-head-gat-lstm-vn.yaml trainer.devices=3 test=false model.backbone.dropout=0.2 model.backbone.heads=4 model.backbone.hidden_dim=64 model.backbone.num_layers=5"
    # (18-28)
    # Excluded "uv run python src/train.py experiment=swh_mwp_mwd_1h_multi-head-gat-lstm-vn.yaml trainer.devices=3 test=false data.max_nodes=250"
    # Excluded "uv run python src/train.py experiment=swh_mwp_mwd_1h_multi-head-gat-lstm-vn.yaml trainer.devices=3 test=false data.max_nodes=250 data.seq_len=3"
    # Excluded "uv run python src/train.py experiment=swh_mwp_mwd_1h_multi-head-gat-lstm-vn.yaml trainer.devices=3 test=false data.max_nodes=250 data.seq_len=12"
    "uv run python src/train.py experiment=swh_mwp_mwd_1h_multi-head-gat-lstm-vn.yaml trainer.devices=3 test=false data.max_nodes=250 data.seq_len=48"
    # Excluded "uv run python src/train.py experiment=swh_mwp_mwd_1h_multi-head-gat-lstm-vn.yaml trainer.devices=3 test=false data.max_nodes=50"
    # Excluded "uv run python src/train.py experiment=swh_mwp_mwd_1h_multi-head-gat-lstm-vn.yaml trainer.devices=3 test=false data.max_nodes=50 data.seq_len=3"
    # Excluded "uv run python src/train.py experiment=swh_mwp_mwd_1h_multi-head-gat-lstm-vn.yaml trainer.devices=3 test=false data.max_nodes=50 data.seq_len=12"
    "uv run python src/train.py experiment=swh_mwp_mwd_1h_multi-head-gat-lstm-vn.yaml trainer.devices=3 test=false data.max_nodes=50 data.seq_len=48"
    # Excluded "uv run python src/train.py experiment=swh_mwp_mwd_1h_multi-head-gat-lstm-vn.yaml trainer.devices=3 test=false data.seq_len=3"
    # Excluded "uv run python src/train.py experiment=swh_mwp_mwd_1h_multi-head-gat-lstm-vn.yaml trainer.devices=3 test=false data.seq_len=12"
    "uv run python src/train.py experiment=swh_mwp_mwd_1h_multi-head-gat-lstm-vn.yaml trainer.devices=3 test=false data.seq_len=48"
    # (29-33)
    "uv run python src/train.py experiment=swh_mwp_mwd_1h_multi-head-gat-lstm-vn.yaml trainer.devices=3 test=false data.seq_len=6"
    "uv run python src/train.py experiment=swh_mwp_mwd_1h_multi-head-gat-lstm-vn.yaml trainer.devices=3 test=false data.seq_len=6 data.max_nodes=250"
    "uv run python src/train.py experiment=swh_mwp_mwd_1h_multi-head-gat-lstm-vn.yaml trainer.devices=3 test=false data.seq_len=6 data.max_nodes=50"
    # Discarded "uv run python src/train.py experiment=swh_mwp_mwd_1h_multi-head-gat.yaml trainer.devices=3 data.max_nodes=50 test=false predict=false"
)

eval "${COMMANDS[$SLURM_ARRAY_TASK_ID]}"

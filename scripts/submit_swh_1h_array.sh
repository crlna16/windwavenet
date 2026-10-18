#!/bin/bash
#SBATCH -A ka1176
#SBATCH -p gpu
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH -o scripts/slurm/slurm-%A_%a.out

# how to call
# sbatch --array=15-18 scripts/submit_swh_1h_array.sh

COMMANDS=(
    # MLP (0)
    "uv run python src/train.py experiment=swh_1h_mlp trainer.devices=1 seed=211100"
    # LSTM (1-5)
    #"uv run python src/train.py experiment=swh_1h_lstm data.seq_len=3  trainer.devices=4 seed=211100"
    "uv run python src/train.py experiment=swh_1h_lstm data.seq_len=6  trainer.devices=4 seed=211100"
    #"uv run python src/train.py experiment=swh_1h_lstm data.seq_len=12 trainer.devices=4 seed=211100"
    #"uv run python src/train.py experiment=swh_1h_lstm data.seq_len=24 trainer.devices=4 seed=211100"
    "uv run python src/train.py experiment=swh_1h_lstm data.seq_len=48 trainer.devices=4 seed=211100"
    # Multi-Head GAT (6-8)
    "uv run python src/train.py experiment=swh_1h_gat.yaml trainer.devices=4 data.max_nodes=500 test=false predict=false seed=211100"
    #"uv run python src/train.py experiment=swh_1h_gat.yaml trainer.devices=4 data.max_nodes=250 test=false predict=false seed=211100"
    #"uv run python src/train.py experiment=swh_1h_gat.yaml trainer.devices=4 data.max_nodes=50  test=false predict=false seed=211100"
    # Multi-Head GAT VN (09-11)
    "uv run python src/train.py experiment=swh_1h_gat-vn.yaml trainer.devices=4 data.max_nodes=500 test=false seed=211100"
    #"uv run python src/train.py experiment=swh_1h_gat-vn.yaml trainer.devices=4 data.max_nodes=250 test=false seed=211100"
    "uv run python src/train.py experiment=swh_1h_gat-vn.yaml trainer.devices=4 data.max_nodes=50 test=false seed=211100"
    # Multi-Head LSTM GAT VN (12-17)
    "uv run python src/train.py experiment=swh_1h_gat-lstm-vn.yaml data.max_nodes=500 data.seq_len=6 trainer.devices=4 test=false seed=211100"
    #"uv run python src/train.py experiment=swh_1h_gat-lstm-vn.yaml data.max_nodes=250 data.seq_len=6 trainer.devices=4 test=false seed=211100"
    "uv run python src/train.py experiment=swh_1h_gat-lstm-vn.yaml data.max_nodes=50  data.seq_len=6 trainer.devices=4 test=false seed=211100"
    "uv run python src/train.py experiment=swh_1h_gat-lstm-vn.yaml data.max_nodes=500 data.seq_len=48 trainer.devices=4 test=false seed=211100"
    #"uv run python src/train.py experiment=swh_1h_gat-lstm-vn.yaml data.max_nodes=250 data.seq_len=48 trainer.devices=4 test=false seed=211100"
    "uv run python src/train.py experiment=swh_1h_gat-lstm-vn.yaml data.max_nodes=50  data.seq_len=48 trainer.devices=4 test=false seed=211100"

)

eval "${COMMANDS[$SLURM_ARRAY_TASK_ID]}"

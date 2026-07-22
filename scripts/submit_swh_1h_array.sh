#!/bin/bash
#SBATCH -A ka1176
#SBATCH -p vader
#SBATCH --gres=gpu:2
#SBATCH --exclusive
#SBATCH --time=48:00:00
#SBATCH -o scripts/slurm/slurm-%A_%a.out
#SBATCH --array=0-34

# how to call
# sbatch --array=15-18 scripts/submit_swh_1h_array.sh

COMMANDS=(
    # DenseNet (0) - TODO
    "uv run python src/train.py experiment=swh_1h_densenet trainer.devices=3"
    # LSTM (1-6) - UPDATED
    "uv run python src/train.py experiment=swh_1h_lstm data.seq_len=3  trainer.devices=2"
    "uv run python src/train.py experiment=swh_1h_lstm data.seq_len=6  trainer.devices=2 'ckpt_path=\"logs/train/swh_1h_lstm/runs/2026-07-22_06-04-45/wwm-emulator-hourly-final/1lt1kqxe/checkpoints/last.ckpt\"' logger.wandb.id=1lt1kqxe"
    "uv run python src/train.py experiment=swh_1h_lstm data.seq_len=12 trainer.devices=2"
    "uv run python src/train.py experiment=swh_1h_lstm data.seq_len=24 trainer.devices=2"
    "uv run python src/train.py experiment=swh_1h_lstm data.seq_len=48 trainer.devices=2"
    # Multi-Head GAT (7-9) - TODO (node)
    # Discarded "uv run python src/train.py experiment=swh_1h_multi-head-gat.yaml trainer.devices=4 data.max_nodes=250 test=false predict=false"
    # Discarded "uv run python src/train.py experiment=swh_1h_multi-head-gat.yaml trainer.devices=4 data.max_nodes=500 test=false predict=false"
    # Discarded "uv run python src/train.py experiment=swh_1h_multi-head-gat.yaml trainer.devices=4 data.max_nodes=1000 test=false predict=false"
    # Multi-Head GAT VN (10-14) - TODO (node)
    "uv run python src/train.py experiment=swh_1h_multi-head-gat-vn.yaml trainer.devices=4 test=false"
)

eval "${COMMANDS[$SLURM_ARRAY_TASK_ID]}"

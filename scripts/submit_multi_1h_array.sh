#!/bin/bash
#SBATCH -A ka1176
#SBATCH -p gpu
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH -o scripts/slurm/slurm-%A_%a.out

# how to call
# sbatch --array=15-18 scripts/submit_multi_1h_array.sh

COMMANDS=(
    # MLP (0)
    # LSTM (1-5)
    "uv run python src/train.py experiment=multi_1h_lstm data.seq_len=6  trainer.devices=4 seed=172426"
    "uv run python src/train.py experiment=multi_1h_lstm data.seq_len=48 trainer.devices=4 seed=172426"
    # Multi-Head GAT (6-8)
    # Multi-Head GAT VN (09-11)
    "uv run python src/train.py experiment=multi_1h_gat-vn.yaml trainer.devices=4 data.max_nodes=500 test=false seed=172426 logger.wandb.id=av9iktbc 'ckpt_path=\"logs/train/multi_1h_gat-vn/runs/2026-10-22_10-30-16/wwm-emulator-hourly-final/av9iktbc/checkpoints/last.ckpt\"'"
    # Multi-Head LSTM GAT VN (12-17)
    "uv run python src/train.py experiment=multi_1h_gat-lstm-vn.yaml data.max_nodes=500 data.seq_len=6 trainer.devices=4 test=false seed=172426 logger.wandb.id=cyrtd2nx 'ckpt_path=\"logs/train/multi_1h_gat-lstm-vn/runs/2026-10-23_00-14-02/wwm-emulator-hourly-final/cyrtd2nx/checkpoints/last.ckpt\"'"
)

eval "${COMMANDS[$SLURM_ARRAY_TASK_ID]}"

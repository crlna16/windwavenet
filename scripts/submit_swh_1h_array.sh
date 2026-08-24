#!/bin/bash
#SBATCH -A ka1176
#SBATCH -p vader
# no#SBATCH --gres=gpu:2
#SBATCH --exclusive
#SBATCH --time=48:00:00
#SBATCH -o scripts/slurm/slurm-%A_%a.out

# how to call
# sbatch --array=15-18 scripts/submit_swh_1h_array.sh

COMMANDS=(
    # MLP (0) - UPDATED
    "uv run python src/train.py experiment=swh_1h_mlp trainer.devices=1"
    # LSTM (1-5) - UPDATED
    "uv run python src/train.py experiment=swh_1h_lstm data.seq_len=3  trainer.devices=2"
    "uv run python src/train.py experiment=swh_1h_lstm data.seq_len=6  trainer.devices=2"
    "uv run python src/train.py experiment=swh_1h_lstm data.seq_len=12 trainer.devices=2"
    "uv run python src/train.py experiment=swh_1h_lstm data.seq_len=24 trainer.devices=2"
    "uv run python src/train.py experiment=swh_1h_lstm data.seq_len=48 trainer.devices=2"
    # Multi-Head GAT (6-8) - INPROGRESS (node)
    "uv run python src/train.py experiment=swh_1h_multi-head-gat.yaml trainer.devices=4 data.max_nodes=500 test=false predict=false"
    "uv run python src/train.py experiment=swh_1h_multi-head-gat.yaml trainer.devices=4 data.max_nodes=250 test=false predict=false"
    "uv run python src/train.py experiment=swh_1h_multi-head-gat.yaml trainer.devices=4 data.max_nodes=50  test=false predict=false"
    # Multi-Head GAT VN (09-11) - INPROGRESS (node)
    "uv run python src/train.py experiment=swh_1h_multi-head-gat-vn.yaml trainer.devices=4 data.max_nodes=500 test=false seed=210856 logger.wandb.id=n2j2gloz 'ckpt_path=\"logs/train/swh_1h_multi-head-gat-vn/runs/2026-08-21_09-57-12/wwm-emulator-hourly-final/n2j2gloz/checkpoints/last.ckpt\"'"
    "uv run python src/train.py experiment=swh_1h_multi-head-gat-vn.yaml trainer.devices=4 data.max_nodes=250 test=false seed=210857 logger.wandb.id=l19h6pc1 'ckpt_path=\"logs/train/swh_1h_multi-head-gat-vn/runs/2026-08-21_09-58-28/wwm-emulator-hourly-final/l19h6pc1/checkpoints/last.ckpt\"'"
    "uv run python src/train.py experiment=swh_1h_multi-head-gat-vn.yaml trainer.devices=4 data.max_nodes=50 test=false seed=210858 logger.wandb.id=2w4mhqs5 'ckpt_path=\"logs/train/swh_1h_multi-head-gat-vn/runs/2026-08-21_13-33-33/wwm-emulator-hourly-final/2w4mhqs5/checkpoints/last.ckpt\"'"
    # Multi-Head LSTM GAT VN (12-17) - TODO
    "uv run python src/train.py experiment=swh_1h_multi-head-gat-lstm-vn.yaml data.max_nodes=500 data.seq_len=6 trainer.devices=4 test=false seed=210813 logger.wandb.id=7gg6p6wq 'ckpt_path=\"./logs/train/swh_1h_multi-head-gat-lstm-vn/runs/2026-08-21_16-53-35/wwm-emulator-hourly-final/7gg6p6wq/checkpoints/last.ckpt\"'"
    "uv run python src/train.py experiment=swh_1h_multi-head-gat-lstm-vn.yaml data.max_nodes=250 data.seq_len=6 trainer.devices=4 test=false seed=210814 logger.wandb.id=21vwuxaz 'ckpt_path=\"logs/train/swh_1h_multi-head-gat-lstm-vn/runs/2026-08-21_21-55-20/wwm-emulator-hourly-final/21vwuxaz/checkpoints/last.ckpt\"'"
    "uv run python src/train.py experiment=swh_1h_multi-head-gat-lstm-vn.yaml data.max_nodes=50  data.seq_len=6 trainer.devices=4 test=false seed=210815"
    "uv run python src/train.py experiment=swh_1h_multi-head-gat-lstm-vn.yaml data.max_nodes=500 data.seq_len=48 trainer.devices=4 test=false seed=210816 logger.wandb.id=s4x4mgnr 'ckpt_path=\"logs/train/swh_1h_multi-head-gat-lstm-vn/runs/2026-08-21_21-58-41/wwm-emulator-hourly-final/s4x4mgnr/checkpoints/last.ckpt\"'"
    "uv run python src/train.py experiment=swh_1h_multi-head-gat-lstm-vn.yaml data.max_nodes=250 data.seq_len=48 trainer.devices=4 test=false seed=210817 logger.wandb.id=laajpx4l 'ckpt_path=\"logs/train/swh_1h_multi-head-gat-lstm-vn/runs/2026-08-22_01-36-10/wwm-emulator-hourly-final/laajpx4l/checkpoints/last.ckpt\"'"
    "uv run python src/train.py experiment=swh_1h_multi-head-gat-lstm-vn.yaml data.max_nodes=50  data.seq_len=48 trainer.devices=4 test=false seed=210818"

)

eval "${COMMANDS[$SLURM_ARRAY_TASK_ID]}"

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
    # LSTM (1-5)
    "uv run python src/train.py experiment=mwd_1h_lstm data.seq_len=6  trainer.devices=2 seed=172426"
    "uv run python src/train.py experiment=mwd_1h_lstm data.seq_len=48 trainer.devices=2 seed=172426"
    "uv run python src/train.py experiment=mwp_1h_lstm data.seq_len=6  trainer.devices=2 seed=172426"
    "uv run python src/train.py experiment=mwp_1h_lstm data.seq_len=48 trainer.devices=2 seed=172426"
    #"uv run python src/train.py experiment=swh_1h_lstm data.seq_len=6  trainer.devices=4 seed=211100 +model.prev_noise_std.sigWaveHeight=0.02"
    #"uv run python src/train.py experiment=swh_1h_lstm data.seq_len=48 trainer.devices=4 seed=211100 +model.prev_noise_std.sigWaveHeight=0.02"
    #"uv run python src/train.py experiment=swh_1h_lstm data.seq_len=6  trainer.devices=4 seed=110814 +model.prev_noise_std.sigWaveHeight=0.02"
    #"uv run python src/train.py experiment=swh_1h_lstm data.seq_len=48 trainer.devices=4 seed=110814 +model.prev_noise_std.sigWaveHeight=0.02"
    # Multi-Head GAT (6-8)
    # Multi-Head GAT VN (09-11)
    "uv run python src/train.py experiment=mwd_1h_gat-vn.yaml trainer.devices=4 data.max_nodes=500 test=false seed=172426"
    "uv run python src/train.py experiment=mwp_1h_gat-vn.yaml trainer.devices=4 data.max_nodes=500 test=false seed=172426 logger.wandb.id=ecyth31q 'ckpt_path=\"logs/train/mwp_1h_gat-vn/runs/2026-10-23_01-01-39/wwm-emulator-hourly-final/ecyth31q/checkpoints/last.ckpt\"'"
    # Multi-Head LSTM GAT VN (12-17)
    "uv run python src/train.py experiment=mwd_1h_gat-lstm-vn.yaml data.max_nodes=500 data.seq_len=6 trainer.devices=4 test=false seed=172426 logger.wandb.id=iyd1b6w4 'ckpt_path=\"logs/train/mwd_1h_gat-lstm-vn/runs/2026-10-23_12-20-46/wwm-emulator-hourly-final/iyd1b6w4/checkpoints/last.ckpt\"'"
    "uv run python src/train.py experiment=mwp_1h_gat-lstm-vn.yaml data.max_nodes=500 data.seq_len=6 trainer.devices=4 test=false seed=172426 logger.wandb.id=edlf2331 'ckpt_path=\"logs/train/mwp_1h_gat-lstm-vn/runs/2026-10-24_14-24-25/wwm-emulator-hourly-final/edlf2331/checkpoints/last.ckpt\"'"

)

eval "${COMMANDS[$SLURM_ARRAY_TASK_ID]}"

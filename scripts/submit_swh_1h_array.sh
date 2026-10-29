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
    "uv run python src/train.py experiment=swh_1h_mlp trainer.devices=1 seed=172426"
    # LSTM (1-5)
    "uv run python src/train.py experiment=swh_1h_lstm data.seq_len=6  trainer.devices=4 seed=211100 +model.prev_noise_std.sigWaveHeight=0.02"
    "uv run python src/train.py experiment=swh_1h_lstm data.seq_len=48 trainer.devices=4 seed=211100 +model.prev_noise_std.sigWaveHeight=0.02"
    "uv run python src/train.py experiment=swh_1h_lstm data.seq_len=6  trainer.devices=4 seed=110814 +model.prev_noise_std.sigWaveHeight=0.02"
    "uv run python src/train.py experiment=swh_1h_lstm data.seq_len=48 trainer.devices=4 seed=110814 +model.prev_noise_std.sigWaveHeight=0.02"
    # Multi-Head GAT (6-8)
    "uv run python src/train.py experiment=swh_1h_gat.yaml trainer.devices=4 data.max_nodes=500 test=false predict=false seed=172426"
    "uv run python src/train.py experiment=swh_1h_gat.yaml trainer.devices=4 data.max_nodes=500 test=false predict=false seed=211100"
    "uv run python src/train.py experiment=swh_1h_gat.yaml trainer.devices=4 data.max_nodes=500 test=false predict=false seed=110814"
    # Multi-Head GAT VN (09-11)
    "uv run python src/train.py experiment=swh_1h_gat-vn.yaml trainer.devices=4 data.max_nodes=500 test=false seed=172426 logger.wandb.id=5gbdho3o 'ckpt_path=\"logs/train/swh_1h_gat-vn/runs/2026-10-19_15-59-30/wwm-emulator-hourly-final/5gbdho3o/checkpoints/last.ckpt\"'"
    "uv run python src/train.py experiment=swh_1h_gat-vn.yaml trainer.devices=4 data.max_nodes=500 test=false seed=211100 logger.wandb.id=4wi6a7vd 'ckpt_path=\"logs/train/swh_1h_gat-vn/runs/2026-10-20_22-20-00/wwm-emulator-hourly-final/4wi6a7vd/checkpoints/last.ckpt\"'"
    "uv run python src/train.py experiment=swh_1h_gat-vn.yaml trainer.devices=4 data.max_nodes=500 test=false seed=110814"
    "uv run python src/train.py experiment=swh_1h_gat-vn.yaml trainer.devices=4 data.max_nodes=50 test=false seed=172426"
    # Multi-Head LSTM GAT VN (12-17)
    "uv run python src/train.py experiment=swh_1h_gat-lstm-vn.yaml data.max_nodes=500 data.seq_len=6 trainer.devices=4 test=false seed=172426 logger.wandb.id=3zahza0f 'ckpt_path=\"logs/train/swh_1h_gat-lstm-vn/runs/2026-10-21_01-19-17/wwm-emulator-hourly-final/3zahza0f/checkpoints/last.ckpt\"'"
    "uv run python src/train.py experiment=swh_1h_gat-lstm-vn.yaml data.max_nodes=500 data.seq_len=6 trainer.devices=4 test=false seed=211100 logger.wandb.id=048pe5np 'ckpt_path=\"logs/train/swh_1h_gat-lstm-vn/runs/2026-10-18_21-14-45/wwm-emulator-hourly-final/048pe5np/checkpoints/last.ckpt\"'"
    "uv run python src/train.py experiment=swh_1h_gat-lstm-vn.yaml data.max_nodes=500 data.seq_len=6 trainer.devices=4 test=false seed=110814 logger.wandb.id=x0rxbsdy 'ckpt_path=\"logs/train/swh_1h_gat-lstm-vn/runs/2026-10-21_01-33-56/wwm-emulator-hourly-final/x0rxbsdy/checkpoints/last.ckpt\"'"
    "uv run python src/train.py experiment=swh_1h_gat-lstm-vn.yaml data.max_nodes=50  data.seq_len=6 trainer.devices=4 test=false seed=172426"
    "uv run python src/train.py experiment=swh_1h_gat-lstm-vn.yaml data.max_nodes=500 data.seq_len=48 trainer.devices=4 test=false seed=172426 logger.wandb.id=gokd7jqx 'ckpt_path=\"logs/train/swh_1h_gat-lstm-vn/runs/2026-10-19_23-23-13/wwm-emulator-hourly-final/gokd7jqx/checkpoints/last.ckpt\"'"
    "uv run python src/train.py experiment=swh_1h_gat-lstm-vn.yaml data.max_nodes=50  data.seq_len=48 trainer.devices=4 test=false seed=172426"
    # Additional GAT VN / LSTM GAT VN with noise and seeds
    "uv run python src/train.py experiment=swh_1h_gat-vn.yaml trainer.devices=4 data.max_nodes=500 test=false seed=172426 +model.prev_noise_std.sigWaveHeight=0.02"
    "uv run python src/train.py experiment=swh_1h_gat-vn.yaml trainer.devices=4 data.max_nodes=500 test=false seed=211100 +model.prev_noise_std.sigWaveHeight=0.02"
    "uv run python src/train.py experiment=swh_1h_gat-vn.yaml trainer.devices=4 data.max_nodes=500 test=false seed=110814 +model.prev_noise_std.sigWaveHeight=0.02"
    #"uv run python src/train.py experiment=swh_1h_gat-lstm-vn.yaml data.max_nodes=500 data.seq_len=6 trainer.devices=4 test=false seed=172426 +model.prev_noise_std.sigWaveHeight=0.02 logger.wandb.id=9ngk8kwt 'ckpt_path=\"logs/train/swh_1h_gat-lstm-vn/runs/2026-10-21_21-39-52/wwm-emulator-hourly-final/9ngk8kwt/checkpoints/epoch=11-step=173928.ckpt\"'"
    "uv run python src/train.py experiment=swh_1h_gat-lstm-vn.yaml data.max_nodes=500 data.seq_len=6 trainer.devices=4 test=false seed=172426 +model.prev_noise_std.sigWaveHeight=0.02 logger.wandb.id=28zljawh 'ckpt_path=\"logs/train/swh_1h_gat-lstm-vn/runs/2026-10-26_00-29-04/wwm-emulator-hourly-final/28zljawh/checkpoints/last.ckpt\"'"
    "uv run python src/train.py experiment=swh_1h_gat-lstm-vn.yaml data.max_nodes=500 data.seq_len=6 trainer.devices=4 test=false seed=211100 +model.prev_noise_std.sigWaveHeight=0.02"
    "uv run python src/train.py experiment=swh_1h_gat-lstm-vn.yaml data.max_nodes=500 data.seq_len=6 trainer.devices=4 test=false seed=110814 +model.prev_noise_std.sigWaveHeight=0.02 logger.wandb.id=7xo1atnw 'ckpt_path=\"logs/train/swh_1h_gat-lstm-vn/runs/2026-10-22_07-03-23/wwm-emulator-hourly-final/7xo1atnw/checkpoints/epoch=10-step=159434.ckpt\"'"
    # Test one more seed
    "uv run python src/train.py experiment=swh_1h_gat-lstm-vn.yaml data.max_nodes=500 data.seq_len=6 trainer.devices=4 test=false seed=251006"
    "uv run python src/train.py experiment=swh_1h_gat-lstm-vn.yaml data.max_nodes=500 data.seq_len=6 trainer.devices=4 test=false seed=251006 +model.prev_noise_std.sigWaveHeight=0.02 logger.wandb.id=f7r7thlh 'ckpt_path=\"logs/train/swh_1h_gat-lstm-vn/runs/2026-10-26_01-30-31/wwm-emulator-hourly-final/f7r7thlh/checkpoints/last.ckpt\"'"

)

eval "${COMMANDS[$SLURM_ARRAY_TASK_ID]}"

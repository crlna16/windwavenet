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
    "uv run python src/train.py experiment=mwp_1h_gat-vn.yaml trainer.devices=4 data.max_nodes=500 test=false seed=172426"
    # Multi-Head LSTM GAT VN (12-17)
    "uv run python src/train.py experiment=mwd_1h_gat-lstm-vn.yaml data.max_nodes=500 data.seq_len=6 trainer.devices=4 test=false seed=172426"
    "uv run python src/train.py experiment=mwp_1h_gat-lstm-vn.yaml data.max_nodes=500 data.seq_len=6 trainer.devices=4 test=false seed=172426"

)

eval "${COMMANDS[$SLURM_ARRAY_TASK_ID]}"

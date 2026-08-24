#!/bin/bash
#SBATCH -A ka1176
#SBATCH -p vader
#SBATCH --mem=200G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH -o scripts/slurm/slurm-%A_%a.out
#SBATCH --array=0-34

# how to call
# sbatch --array=15-18 scripts/submit_swh_1h_array.sh

COMMANDS=(
    # DenseNet (0)
    "uv run python src/predict.py experiment=swh_1h_densenet logger.wandb.id=9m79pxc0 'ckpt_path=\"logs/train/swh_1h_densenet/runs/2026-07-23_09-53-43/wwm-emulator-hourly-final/9m79pxc0/checkpoints/epoch=51-step=188812.ckpt\"'"
    # LSTM (1-5)
    # TODO "uv run python src/predict.py --config-name=predict_lstm experiment=swh_1h_lstm data.seq_len=3
    # TODO "uv run python src/predict.py --config-name=predict_lstm experiment=swh_1h_lstm data.seq_len=6
    # TODO "uv run python src/predict.py --config-name=predict_lstm experiment=swh_1h_lstm data.seq_len=12
    # TODO "uv run python src/predict.py --config-name=predict_lstm experiment=swh_1h_lstm data.seq_len=24
    # TODO "uv run python src/predict.py --config-name=predict_lstm experiment=swh_1h_lstm data.seq_len=48
    # TODO # Multi-Head GAT (6-8)
    # TODO "uv run python src/predict_gcn.py experiment=swh_1h_multi-head-gat.yaml data.max_nodes=500
    # TODO "uv run python src/predict_gcn.py experiment=swh_1h_multi-head-gat.yaml data.max_nodes=250
    # TODO "uv run python src/predict_gcn.py experiment=swh_1h_multi-head-gat.yaml data.max_nodes=50
    # TODO # Multi-Head GAT VN (9-11)
    # TODO "uv run python src/predict_gcn.py experiment=swh_1h_multi-head-gat-vn.yaml data.max_nodes=500
    # TODO "uv run python src/predict_gcn.py experiment=swh_1h_multi-head-gat-vn.yaml data.max_nodes=250
    # TODO "uv run python src/predict_gcn.py experiment=swh_1h_multi-head-gat-vn.yaml data.max_nodes=50
    # TODO # Multi-Head GAT LSTM VN (12-17)
    # TODO "uv run python src/predict_gcn.py experiment=swh_1h_multi-head-gat-lstm-vn.yaml data.seq_len=6 data.max_nodes=500
    # TODO "uv run python src/predict_gcn.py experiment=swh_1h_multi-head-gat-lstm-vn.yaml data.seq_len=6 data.max_nodes=250
    # TODO "uv run python src/predict_gcn.py experiment=swh_1h_multi-head-gat-lstm-vn.yaml data.seq_len=6 data.max_nodes=50
    # TODO "uv run python src/predict_gcn.py experiment=swh_1h_multi-head-gat-lstm-vn.yaml data.seq_len=48 data.max_nodes=500
    # TODO "uv run python src/predict_gcn.py experiment=swh_1h_multi-head-gat-lstm-vn.yaml data.seq_len=48 data.max_nodes=250
    # TODO "uv run python src/predict_gcn.py experiment=swh_1h_multi-head-gat-lstm-vn.yaml data.seq_len=48 data.max_nodes=50
)

eval "${COMMANDS[$SLURM_ARRAY_TASK_ID]}"

#!/bin/bash
#SBATCH -A ka1176
#SBATCH -p vader
#SBATCH --mem=200G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH -o scripts/slurm/slurm-%A_%a.out
#SBATCH --array=0-34

# how to call
# sbatch --array=15-18 scripts/submit_swh_mwp_mwd_1h_array.sh

COMMANDS=(
    # DenseNet (0)
    "uv run python src/predict.py experiment=swh_mwp_mwd_1h_densenet 'ckpt_path=\"./logs/train/swh_mwp_mwd_1h_densenet/runs/2026-06-26_14-01-49/wwm-emulator-hourly-final/ucf0lwmp/checkpoints/epoch=2-step=5298.ckpt\"' 'logger.wandb.id=ucf0lwmp'"
    # LSTM (1-5)
    "uv run python src/predict.py --config-name=predict_lstm experiment=swh_mwp_mwd_1h_lstm data.seq_len=3 'ckpt_path=\"./logs/train/swh_mwp_mwd_1h_lstm/runs/2026-06-26_14-01-49/wwm-emulator-hourly-final/9bsyvfh6/checkpoints/epoch=30-step=41013.ckpt\"' logger.wandb.id='9bsyvfh6'"
    "uv run python src/predict.py --config-name=predict_lstm experiment=swh_mwp_mwd_1h_lstm data.seq_len=6 'ckpt_path=\"./logs/train/swh_mwp_mwd_1h_lstm/runs/2026-06-26_14-01-49/wwm-emulator-hourly-final/o4zgpyxn/checkpoints/epoch=24-step=33025.ckpt\"' 'logger.wandb.id=o4zgpyxn'"
    "uv run python src/predict.py --config-name=predict_lstm experiment=swh_mwp_mwd_1h_lstm data.seq_len=12 'ckpt_path=\"./logs/train/swh_mwp_mwd_1h_lstm/runs/2026-06-26_14-16-59/wwm-emulator-hourly-final/cs3k5sd7/checkpoints/epoch=28-step=38222.ckpt\"' 'logger.wandb.id=cs3k5sd7'"
    "uv run python src/predict.py --config-name=predict_lstm experiment=swh_mwp_mwd_1h_lstm data.seq_len=24 'ckpt_path=\"./logs/train/swh_mwp_mwd_1h_lstm/runs/2026-06-26_14-27-20/wwm-emulator-hourly-final/8kvtqq9v/checkpoints/epoch=28-step=37990.ckpt\"' 'logger.wandb.id=8kvtqq9v'"
    "uv run python src/predict.py --config-name=predict_lstm experiment=swh_mwp_mwd_1h_lstm data.seq_len=48 'ckpt_path=\"./logs/train/swh_mwp_mwd_1h_lstm/runs/2026-06-26_14-27-20/wwm-emulator-hourly-final/9jg5gjtp/checkpoints/epoch=32-step=42735.ckpt\"' 'logger.wandb.id=9jg5gjtp'"
    # Multi-Head GAT ()
    # Multi-Head GAT VN (6-8)
    "uv run python src/predict_gcn.py experiment=swh_mwp_mwd_1h_multi-head-gat-vn.yaml data.max_nodes=50 'ckpt_path=\"./logs/train/swh_mwp_mwd_1h_multi-head-gat-vn/runs/2026-06-29_17-36-53/wwm-emulator-hourly-final/89lo49ce/checkpoints/epoch=8-step=127107.ckpt\"' 'logger.wandb.id=89lo49ce'"
    "uv run python src/predict_gcn.py experiment=swh_mwp_mwd_1h_multi-head-gat-vn.yaml data.max_nodes=250 'ckpt_path=\"./logs/train/swh_mwp_mwd_1h_multi-head-gat-vn/runs/2026-06-29_17-38-46/wwm-emulator-hourly-final/40ijyt36/checkpoints/epoch=12-step=183599.ckpt\"' 'logger.wandb.id=40ijyt36'"
    "uv run python src/predict_gcn.py experiment=swh_mwp_mwd_1h_multi-head-gat-vn.yaml data.max_nodes=500 'ckpt_path=\"./logs/train/swh_mwp_mwd_1h_multi-head-gat-vn/runs/2026-06-29_17-36-53/wwm-emulator-hourly-final/clilfc91/checkpoints/epoch=10-step=155353.ckpt\"' 'logger.wandb.id=clilfc91'"
    # Multi-Head GAT LSTM VN (9-14)
    "uv run python src/predict_gcn.py experiment=swh_mwp_mwd_1h_multi-head-gat-lstm-vn.yaml data.seq_len=6 data.max_nodes=50 'ckpt_path=\"./logs/train/swh_mwp_mwd_1h_multi-head-gat-lstm-vn/runs/2026-06-30_01-39-40/wwm-emulator-hourly-final/3z6pvxt8/checkpoints/epoch=14-step=211350.ckpt\"' 'logger.wandb.id=3z6pvxt8'"
    "uv run python src/predict_gcn.py experiment=swh_mwp_mwd_1h_multi-head-gat-lstm-vn.yaml data.seq_len=6 data.max_nodes=250 'ckpt_path=\"./logs/train/swh_mwp_mwd_1h_multi-head-gat-lstm-vn/runs/2026-06-30_01-37-41/wwm-emulator-hourly-final/g6r8gliu/checkpoints/epoch=16-step=239530.ckpt\"' 'logger.wandb.id=g6r8gliu'"
    "uv run python src/predict_gcn.py experiment=swh_mwp_mwd_1h_multi-head-gat-lstm-vn.yaml data.seq_len=6 'ckpt_path=\"./logs/train/swh_mwp_mwd_1h_multi-head-gat-lstm-vn/runs/2026-06-29_23-45-41/wwm-emulator-hourly-final/ne5nhxkm/checkpoints/epoch=9-step=140900.ckpt\"' 'logger.wandb.id=ne5nhxkm'"
    "uv run python src/predict_gcn.py experiment=swh_mwp_mwd_1h_multi-head-gat-lstm-vn.yaml data.max_nodes=50 data.seq_len=48 'ckpt_path=\"./logs/train/swh_mwp_mwd_1h_multi-head-gat-lstm-vn/runs/2026-06-29_19-23-07/wwm-emulator-hourly-final/38tds33a/checkpoints/epoch=19-step=276240.ckpt\"' 'logger.wandb.id=38tds33a'"
    "uv run python src/predict_gcn.py experiment=swh_mwp_mwd_1h_multi-head-gat-lstm-vn.yaml data.max_nodes=250 data.seq_len=48 'ckpt_path=\"./logs/train/swh_mwp_mwd_1h_multi-head-gat-lstm-vn/runs/2026-06-29_17-39-09/wwm-emulator-hourly-final/5drgmwb0/checkpoints/epoch=13-step=193368.ckpt\"' 'logger.wandb.id=5drgmwb0'"
    "uv run python src/predict_gcn.py experiment=swh_mwp_mwd_1h_multi-head-gat-lstm-vn.yaml data.seq_len=48 'ckpt_path=\"./logs/train/swh_mwp_mwd_1h_multi-head-gat-lstm-vn/runs/2026-06-29_21-26-43/wwm-emulator-hourly-final/qgxfz6dx/checkpoints/epoch=11-step=165744.ckpt\"' 'logger.wandb.id=qgxfz6dx'"
)

# Stagger job starts by 1 minute per array index to avoid simultaneous output-dir creation.
sleep $((SLURM_ARRAY_TASK_ID * 60))

eval "${COMMANDS[$SLURM_ARRAY_TASK_ID]}"

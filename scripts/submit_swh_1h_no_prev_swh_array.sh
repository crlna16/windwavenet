#!/bin/bash
#SBATCH -A ka1176
#SBATCH -p vader
# no#SBATCH --gres=gpu:2
#SBATCH --exclusive
#SBATCH --time=48:00:00
#SBATCH -o scripts/slurm/slurm-%A_%a.out

# Ablation: same experiments/hyperparameters as submit_swh_1h_array.sh, but
# "prev_sigWaveHeight" is dropped from data.features, so the backbone
# input_dim must drop from 7 to 6. These are fresh runs (no ckpt_path resume)
# since the removed feature changes the backbone's input layer shape.

# how to call
# sbatch --array=0-17 scripts/submit_swh_1h_no_prev_swh_array.sh

FEATURES_NO_PREV_SWH="[windSpeedX,windSpeedY,elevation,depthAverageVelX,depthAverageVelY,peakPeriod]"

COMMANDS=(
    # MLP (0)
    "uv run python src/train.py experiment=swh_1h_mlp trainer.devices=1 data.features=$FEATURES_NO_PREV_SWH model.backbone.input_dim=6 experiment_name=swh_1h_mlp_no_prev_swh logger.wandb.tags=[swh-1h,densenet,no-prev]"
    # LSTM (1-2)
    "uv run python src/train.py experiment=swh_1h_lstm data.seq_len=6  trainer.devices=1 data.features=$FEATURES_NO_PREV_SWH model.backbone.input_dim=6 experiment_name=swh_1h_lstm_no_prev_swh logger.wandb.tags=[swh-1h,lstm,no-prev]"
    "uv run python src/train.py experiment=swh_1h_lstm data.seq_len=48 trainer.devices=1 data.features=$FEATURES_NO_PREV_SWH model.backbone.input_dim=6 experiment_name=swh_1h_lstm_no_prev_swh logger.wandb.tags=[swh-1h,lstm,no-prev]"
    # Multi-Head GAT (3-4)
    "uv run python src/train.py experiment=swh_1h_multi-head-gat.yaml trainer.devices=4 data.max_nodes=500 test=false predict=false data.features=$FEATURES_NO_PREV_SWH model.backbone.input_dim=6 experiment_name=swh_1h_multi-head-gat_no_prev_swh logger.wandb.tags=[swh-1h,gat,no-prev]"
    "uv run python src/train.py experiment=swh_1h_multi-head-gat.yaml trainer.devices=4 data.max_nodes=50  test=false predict=false data.features=$FEATURES_NO_PREV_SWH model.backbone.input_dim=6 experiment_name=swh_1h_multi-head-gat_no_prev_swh logger.wandb.tags=[swh-1h,gat,no-prev]"
    # Multi-Head GAT VN (5-6)
    "uv run python src/train.py experiment=swh_1h_multi-head-gat-vn.yaml trainer.devices=4 data.max_nodes=500 test=false seed=210856 data.features=$FEATURES_NO_PREV_SWH model.backbone.input_dim=6 experiment_name=swh_1h_multi-head-gat-vn_no_prev_swh logger.wandb.tags=[swh-1h,gat-vn,no-prev]"
    "uv run python src/train.py experiment=swh_1h_multi-head-gat-vn.yaml trainer.devices=4 data.max_nodes=50 test=false seed=210858 data.features=$FEATURES_NO_PREV_SWH model.backbone.input_dim=6 experiment_name=swh_1h_multi-head-gat-vn_no_prev_swh logger.wandb.tags=[swh-1h,gat-vn,no-prev]"
    # Multi-Head LSTM GAT VN (7-10)
    "uv run python src/train.py experiment=swh_1h_multi-head-gat-lstm-vn.yaml data.max_nodes=500 data.seq_len=6 trainer.devices=4 test=false seed=210813 data.features=$FEATURES_NO_PREV_SWH model.backbone.input_dim=6 experiment_name=swh_1h_multi-head-gat-lstm-vn_no_prev_swh logger.wandb.tags=[swh-1h,gat-lstm-vn,no-prev]"
    "uv run python src/train.py experiment=swh_1h_multi-head-gat-lstm-vn.yaml data.max_nodes=50  data.seq_len=6 trainer.devices=4 test=false seed=210815 data.features=$FEATURES_NO_PREV_SWH model.backbone.input_dim=6 experiment_name=swh_1h_multi-head-gat-lstm-vn_no_prev_swh logger.wandb.tags=[swh-1h,gat-lstm-vn,no-prev]"
    "uv run python src/train.py experiment=swh_1h_multi-head-gat-lstm-vn.yaml data.max_nodes=500 data.seq_len=48 trainer.devices=4 test=false seed=210816 data.features=$FEATURES_NO_PREV_SWH model.backbone.input_dim=6 experiment_name=swh_1h_multi-head-gat-lstm-vn_no_prev_swh logger.wandb.tags=[swh-1h,gat-lstm-vn,no-prev]"
    "uv run python src/train.py experiment=swh_1h_multi-head-gat-lstm-vn.yaml data.max_nodes=50  data.seq_len=48 trainer.devices=4 test=false seed=210818 data.features=$FEATURES_NO_PREV_SWH model.backbone.input_dim=6 experiment_name=swh_1h_multi-head-gat-lstm-vn_no_prev_swh logger.wandb.tags=[swh-1h,gat-lstm-vn,no-prev]"

)

eval "${COMMANDS[$SLURM_ARRAY_TASK_ID]}"

#!/bin/bash
#SBATCH -A ka1176
#SBATCH -p gpu
#SBATCH --mem=200G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH -o scripts/slurm/slurm-%A_%a.out
#SBATCH --array=0-34

# how to call
# sbatch --array=15-18 scripts/submit_swh_1h_array.sh

COMMANDS=(
    # DenseNet (0)
    "uv run python src/predict.py experiment=swh_1h_densenet 'ckpt_path=\"/work/ka1176/caroline/gitlab/wave-wind-model-emulator/logs/train/shw_1h_densenet/runs/2026-06-12_13-08-59/wwm-emulator-hourly-final/vlu7fztd/checkpoints/epoch=2-step=3972.ckpt\"' 'logger.wandb.id=vlu7fztd'"
    # LSTM (1-5)
    "uv run python src/predict.py --config-name=predict_lstm experiment=swh_1h_lstm data.seq_len=3 'ckpt_path=\"logs/train/shw_1h_lstm/runs/2026-06-12_13-14-22/wwm-emulator-hourly-final/she1j3a6/checkpoints/epoch=33-step=44982.ckpt\"' 'logger.wandb.id=she1j3a6'"
    "uv run python src/predict.py --config-name=predict_lstm experiment=swh_1h_lstm data.seq_len=6 'ckpt_path=\"logs/train/shw_1h_lstm/runs/2026-06-12_13-14-23/wwm-emulator-hourly-final/psddgmra/checkpoints/epoch=34-step=46235.ckpt\"' 'logger.wandb.id=psddgmra'"
    "uv run python src/predict.py --config-name=predict_lstm experiment=swh_1h_lstm data.seq_len=12 'ckpt_path=\"logs/train/shw_1h_lstm/runs/2026-06-12_13-14-25/wwm-emulator-hourly-final/famt9uco/checkpoints/epoch=22-step=30314.ckpt\"' 'logger.wandb.id=famt9uco'"
    "uv run python src/predict.py --config-name=predict_lstm experiment=swh_1h_lstm data.seq_len=24 'ckpt_path=\"./logs/train/shw_1h_lstm/runs/2026-06-12_13-16-19/wwm-emulator-hourly-final/rncpe3y2/checkpoints/epoch=24-step=32750.ckpt\"' 'logger.wandb.id=rncpe3y2'"
    "uv run python src/predict.py --config-name=predict_lstm experiment=swh_1h_lstm data.seq_len=48 'ckpt_path=\"logs/train/shw_1h_lstm/runs/2026-06-12_13-25-10/wwm-emulator-hourly-final/gnghqhwv/checkpoints/epoch=24-step=32375.ckpt\"' 'logger.wandb.id=gnghqhwv'"
    # Multi-Head GAT ()
    # Multi-Head GAT VN (6-8)
    "uv run python src/predict_gcn.py experiment=swh_1h_multi-head-gat-vn.yaml 'ckpt_path=\"./logs/train/swh_1h_multi-head-gat-vn/runs/2026-06-18_14-30-33/wwm-emulator-hourly-final/gr3xp6ql/checkpoints/epoch=11-step=127104.ckpt\"' 'logger.wandb.id=gr3xp6ql'"
    "uv run python src/predict_gcn.py experiment=swh_1h_multi-head-gat-vn.yaml data.max_nodes=50 'ckpt_path=\"logs/train/swh_1h_multi-head-gat-vn/runs/2026-06-12_15-32-05/wwm-emulator-hourly-final/sc7ctm6d/checkpoints/epoch=8-step=95328.ckpt\"' 'logger.wandb.id=sc7ctm6d'"
    "uv run python src/predict_gcn.py experiment=swh_1h_multi-head-gat-vn.yaml data.max_nodes=250 'ckpt_path=\"logs/train/swh_1h_multi-head-gat-vn/runs/2026-06-18_14-30-33/wwm-emulator-hourly-final/yoh4zask/checkpoints/epoch=18-step=201248.ckpt\"' 'logger.wandb.id=yoh4zask'"
    # Multi-Head GAT LSTM VN (9-14)
    "uv run python src/predict_gcn.py experiment=swh_1h_multi-head-gat-lstm-vn.yaml data.seq_len=6 data.max_nodes=50 'ckpt_path=\"./logs/train/swh_1h_multi-head-gat-lstm-vn/runs/2026-06-18_12-26-41/wwm-emulator-hourly-final/jgk32o4k/checkpoints/epoch=9-step=105670.ckpt\"' 'logger.wandb.id=jgk32o4k'"
    "uv run python src/predict_gcn.py experiment=swh_1h_multi-head-gat-lstm-vn.yaml data.seq_len=6 data.max_nodes=250 'ckpt_path=\"./logs/train/swh_1h_multi-head-gat-lstm-vn/runs/2026-06-18_12-26-41/wwm-emulator-hourly-final/m8bav13r/checkpoints/epoch=8-step=95103.ckpt\"' 'logger.wandb.id=m8bav13r'"
    "uv run python src/predict_gcn.py experiment=swh_1h_multi-head-gat-lstm-vn.yaml data.seq_len=6 'ckpt_path=\"./logs/train/swh_1h_multi-head-gat-lstm-vn/runs/2026-06-18_12-26-41/wwm-emulator-hourly-final/jg59ucn3/checkpoints/epoch=8-step=95103.ckpt\"' 'logger.wandb.id=jg59ucn3'"
    "uv run python src/predict_gcn.py experiment=swh_1h_multi-head-gat-lstm-vn.yaml data.max_nodes=50 data.seq_len=48 'ckpt_path=\"./logs/train/swh_1h_multi-head-gat-vn/runs/2026-06-13_05-31-00/wwm-emulator-hourly-final/t2likicm/checkpoints/epoch=7-step=82872.ckpt\"' 'logger.wandb.id=t2likicm'"
    "uv run python src/predict_gcn.py experiment=swh_1h_multi-head-gat-lstm-vn.yaml data.max_nodes=250 data.seq_len=48 'ckpt_path=\"./logs/train/swh_1h_multi-head-gat-vn/runs/2026-06-13_03-09-01/wwm-emulator-hourly-final/97vhjpjp/checkpoints/epoch=9-step=103590.ckpt\"' 'logger.wandb.id=97vhjpjp'"
    "uv run python src/predict_gcn.py experiment=swh_1h_multi-head-gat-lstm-vn.yaml data.seq_len=48 'ckpt_path=\"./logs/train/swh_1h_multi-head-gat-vn/runs/2026-06-13_08-07-43/wwm-emulator-hourly-final/69n8g8mm/checkpoints/epoch=2-step=31077.ckpt\"' 'logger.wandb.id=69n8g8mm'"
)

eval "${COMMANDS[$SLURM_ARRAY_TASK_ID]}"

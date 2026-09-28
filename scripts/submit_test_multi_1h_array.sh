#!/bin/bash
#SBATCH -A ka1176
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH -o scripts/slurm/slurm-%A_%a.out

# how to call
# sbatch --array=15-18 scripts/submit_multi_1h_array.sh

COMMANDS=(
    # MLP (0)
    # LSTM (1-5)
    "uv run python src/predict.py --config-name predict_lstm experiment=multi_1h_lstm data.seq_len=6 paths.output_dir='/work/ka1176/caroline/gitlab/wind-wave-model-emulator/logs/predict_final/multi-lstm-6' logger.wandb.id=gvqgjxyn 'ckpt_path=\"logs/train/multi_1h_lstm/runs/2026-09-14_18-17-11/wwm-emulator-hourly-final/gvqgjxyn/checkpoints/epoch=94-step=344280.ckpt\"'"
    "uv run python src/predict.py --config-name predict_lstm experiment=multi_1h_lstm data.seq_len=48 node_chunk_size=32768 paths.output_dir='/work/ka1176/caroline/gitlab/wind-wave-model-emulator/logs/predict_final/multi-lstm-48' logger.wandb.id=p28mshn4 'ckpt_path=\"logs/train/multi_1h_lstm/runs/2026-09-15_06-25-05/wwm-emulator-hourly-final/p28mshn4/checkpoints/epoch=29-step=107160.ckpt\"'"
    # Multi-Head GAT (6-8)
    # Multi-Head GAT VN (09-11)
    "uv run python src/predict_gcn.py experiment=multi_1h_gat-vn.yaml data.max_nodes=500 paths.output_dir='/work/ka1176/caroline/gitlab/wind-wave-model-emulator/logs/predict_final/multi-gat-vn-500' logger.wandb.id=yst7893t 'ckpt_path=\"logs/train/multi_1h_gat-vn/runs/2026-09-15_21-36-38/wwm-emulator-hourly-final/yst7893t/checkpoints/epoch=18-step=275899.ckpt\"'"
    # Multi-Head LSTM GAT VN (12-17)
    "uv run python src/predict_gcn.py experiment=multi_1h_gat-lstm-vn.yaml data.max_nodes=500 data.seq_len=6 paths.output_dir='/work/ka1176/caroline/gitlab/wind-wave-model-emulator/logs/predict_final/multi-gat-lstm-vn-6-500' logger.wandb.id=6flusa96 'ckpt_path=\"logs/train/multi_1h_gat-lstm-vn/runs/2026-09-15_21-36-38/wwm-emulator-hourly-final/6flusa96/checkpoints/epoch=21-step=318912.ckpt\"'"

)

eval "${COMMANDS[$SLURM_ARRAY_TASK_ID]}"

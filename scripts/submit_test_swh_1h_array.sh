#!/bin/bash
#SBATCH -A ka1176
#SBATCH -p gpu
#SBATCH --mem=200G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH -o scripts/slurm/slurm-%A_%a.out

# how to call
# sbatch --array=15-18 scripts/submit_swh_1h_array.sh

COMMANDS=(
    # DenseNet (0)
    "uv run python src/predict.py experiment=swh_1h_mlp paths.output_dir='/work/ka1176/caroline/gitlab/wave-wind-model-emulator/logs/predict_final/swh-mlp' logger.wandb.id=sytiiae9 'ckpt_path=\"logs/train/swh_1h_densenet/runs/2026-09-10_14-22-52/wwm-emulator-hourly-final/sytiiae9/checkpoints/epoch=58-step=214229.ckpt\"'"
    # LSTM (1-2)
    "uv run python src/predict.py --config-name=predict_lstm experiment=swh_1h_lstm data.seq_len=6 paths.output_dir='/work/ka1176/caroline/gitlab/wave-wind-model-emulator/logs/predict_final/swh-lstm-6' logger.wandb.id=k6ydmtnc 'ckpt_path=\"logs/train/swh_1h_lstm/runs/2026-09-10_14-22-52/wwm-emulator-hourly-final/k6ydmtnc/checkpoints/epoch=41-step=152208.ckpt\"'"
    "uv run python src/predict.py --config-name=predict_lstm experiment=swh_1h_lstm data.seq_len=48 node_chunk_size=32768 paths.output_dir='/work/ka1176/caroline/gitlab/wave-wind-model-emulator/logs/predict_final/swh-lstm-48' logger.wandb.id=0pq97b9h 'ckpt_path=\"logs/train/swh_1h_lstm/runs/2026-09-10_14-22-52/wwm-emulator-hourly-final/0pq97b9h/checkpoints/epoch=23-step=85728.ckpt\"'"
    # Multi-Head GAT (3)
    "uv run python src/predict_gcn.py experiment=swh_1h_gat.yaml data.max_nodes=500 paths.output_dir='/work/ka1176/caroline/gitlab/wave-wind-model-emulator/logs/predict_final/swh-gat-500' logger.wandb.id=zwu7keey 'ckpt_path=\"logs/train/swh_1h_gat/runs/2026-09-10_14-22-52/wwm-emulator-hourly-final/zwu7keey/checkpoints/epoch=5-step=87126.ckpt\"'"
    # Multi-Head GAT VN (4-5)
    "uv run python src/predict_gcn.py experiment=swh_1h_gat-vn.yaml data.max_nodes=500 paths.output_dir='/work/ka1176/caroline/gitlab/wave-wind-model-emulator/logs/predict_final/swh-gat-vn-500' TODO"
    "uv run python src/predict_gcn.py experiment=swh_1h_gat-vn.yaml data.max_nodes=50 paths.output_dir='/work/ka1176/caroline/gitlab/wave-wind-model-emulator/logs/predict_final/swh-gat-vn-50' logger.wandb.id=k6kbc2nd 'ckpt_path=\"logs/train/swh_1h_gat-vn/runs/2026-09-10_19-03-48/wwm-emulator-hourly-final/k6kbc2nd/checkpoints/epoch=25-step=377546.ckpt\"'"
    # Multi-Head GAT LSTM VN (6-9)
    "uv run python src/predict_gcn.py experiment=swh_1h_gat-lstm-vn.yaml data.seq_len=6 data.max_nodes=500 paths.output_dir='/work/ka1176/caroline/gitlab/wave-wind-model-emulator/logs/predict_final/swh-gat-lstm-vn-6-500' TODO"
    "uv run python src/predict_gcn.py experiment=swh_1h_gat-lstm-vn.yaml data.seq_len=6 data.max_nodes=50 paths.output_dir='/work/ka1176/caroline/gitlab/wave-wind-model-emulator/logs/predict_final/swh-gat-lstm-vn-6-50' TODO"
    "uv run python src/predict_gcn.py experiment=swh_1h_gat-lstm-vn.yaml data.seq_len=48 data.max_nodes=500 paths.output_dir='/work/ka1176/caroline/gitlab/wave-wind-model-emulator/logs/predict_final/swh-gat-lstm-vn-48-500' TODO"
    "uv run python src/predict_gcn.py experiment=swh_1h_gat-lstm-vn.yaml data.seq_len=48 data.max_nodes=50 paths.output_dir='/work/ka1176/caroline/gitlab/wave-wind-model-emulator/logs/predict_final/swh-gat-lstm-vn-48-50' TODO"
    # END
)

eval "${COMMANDS[$SLURM_ARRAY_TASK_ID]}"

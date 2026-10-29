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
    "uv run python src/predict.py --config-name predict_lstm experiment=wwnet2_lstm data.test_days='[183,213]' paths.output_dir='/work/ka1176/caroline/gitlab/wind-wave-model-emulator/logs/predict_final_wwnet2_SNS00a/wwnet2_lstm' logger.wandb.id=9lwfjpib 'ckpt_path=\"logs/train/wwnet2_lstm/runs/2026-10-14_14-03-21/windwavenet2/9lwfjpib/checkpoints/epoch=47-step=89136.ckpt\"'"
    "uv run python src/predict_gcn.py experiment=wwnet2_graph-lstm data.test_days='[183,213]' paths.output_dir='/work/ka1176/caroline/gitlab/wind-wave-model-emulator/logs/predict_final_wwnet2_SNS00a/wwnet2_graph-lstm' logger.wandb.id=irq81hsx  'ckpt_path=\"logs/train/wwnet2_graph-lstm/runs/2026-10-15_09-52-11/windwavenet2/irq81hsx/checkpoints/epoch=20-step=155925.ckpt\"'"

)

eval "${COMMANDS[$SLURM_ARRAY_TASK_ID]}"

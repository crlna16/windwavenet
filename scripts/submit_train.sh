#!/bin/bash
#SBATCH -A ka1176
#SBATCH -p gpu
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH -o scripts/slurm/slurm-%j.out


uv run python src/train.py experiment=swh_1h_gat-lstm-vn.yaml data.max_nodes=500 data.seq_len=6 trainer.devices=4 test=false seed=211100 +model.prev_noise_std.sigWaveHeight=0.02 logger.wandb.id=6t7bjluh 'ckpt_path="logs/train/swh_1h_gat-lstm-vn/runs/2026-10-27_14-15-07/wwm-emulator-hourly-final/6t7bjluh/checkpoints/last.ckpt"'

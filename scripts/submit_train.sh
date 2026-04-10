#!/bin/bash
#SBATCH -A ka1176
#SBATCH -p vader
#SBATCH --mem=500G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH -o scripts/slurm/slurm-%j.out


uv run src/train.py --config-name=train_gcn data.batch_size=256 "data.train_days=[1, 90]" data.node_idx_csv=src/data/node_selection/node_idx_closest_center_10000.csv test=false model.backbone.hidden_dim=64 model.backbone.num_layers=20

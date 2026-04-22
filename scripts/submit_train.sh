#!/bin/bash
#SBATCH -A ka1176
#SBATCH -p vader
#SBATCH --mem=500G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH -o scripts/slurm/slurm-%j.out


#uv run src/train.py --config-name=train_gcn data.batch_size=256 "data.train_days=[1, 90]" data.node_idx_csv=src/data/node_selection/node_idx_closest_center_10000.csv test=false experiment=gcn_epd

#uv run src/train.py --config-name=train_gcn_epd data.batch_size=256 data.max_links=10000000 data.max_nodes=1000 data.max_radius=200000. model.backbone.decoder_layers=2 model.backbone.dropout=0.1 model.backbone.encoder_layers=2 model.backbone.hidden_dim=32 model.backbone.num_mp_layers=10 model.backbone.use_residual=True model.optimizer.lr=0.0007 experiment=gcn_1h callbacks.early_stopping.patience=3 model.scheduler.patience=2 "data.train_days=[1, 90]" trainer.enable_progress_bar=false test=true predict=false data.scaling_config=src/data/scaling/scale_2020_JFM_hourly.yaml data.node_idx_csv=src/data/node_selection/node_idx_closest_center_10000.csv data.edge_weight_scale=10000

#uv run src/train.py --config-name=train_gcn_lstm data.batch_size=256 data.max_links=10000000 data.max_nodes=1000 data.max_radius=200000. model.optimizer.lr=0.0007 experiment=gcn_1h callbacks.early_stopping.patience=3 model.scheduler.patience=2 "data.train_days=[1, 90]" trainer.enable_progress_bar=false test=false predict=false data.scaling_config=src/data/scaling/scale_2020_JFM_hourly.yaml data.node_idx_csv=src/data/node_selection/node_idx_closest_center_10000.csv
#uv run src/train.py --config-name=train_gcn_lstm data.seq_len=3 data.batch_size=256 data.max_links=10000000 data.max_nodes=1000 data.max_radius=200000. model.optimizer.lr=0.0007 experiment=gcn_1h callbacks.early_stopping.patience=3 model.scheduler.patience=2 "data.train_days=[1, 90]" trainer.enable_progress_bar=false test=false predict=false data.scaling_config=src/data/scaling/scale_2020_JFM_hourly.yaml data.node_idx_csv=src/data/node_selection/node_idx_closest_center_10000.csv

uv run src/train.py --config-name=train_gcn experiment=gcn_1h data.batch_size=256 test=false predict=false  "data.train_days=[1, 90]" data.node_idx_csv=src/data/node_selection/node_idx_closest_center_10000.csv model.focal_gamma=1.0

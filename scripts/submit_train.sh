#!/bin/bash
#SBATCH -A ka1176
#SBATCH -p vader
#SBATCH --mem=200G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH -o scripts/slurm/slurm-%j.out


#uv run src/train.py --config-name=train_gcn data.batch_size=256 "data.train_days=[1, 90]" data.node_idx_csv=src/data/node_selection/node_idx_closest_center_10000.csv test=false experiment=gcn_epd

#uv run src/train.py --config-name=train_gcn_epd data.batch_size=256 data.max_links=10000000 data.max_nodes=1000 data.max_radius=200000. model.backbone.decoder_layers=2 model.backbone.dropout=0.1 model.backbone.encoder_layers=2 model.backbone.hidden_dim=32 model.backbone.num_mp_layers=10 model.backbone.use_residual=True model.optimizer.lr=0.0007 experiment=gcn_1h callbacks.early_stopping.patience=3 model.scheduler.patience=2 "data.train_days=[1, 90]" trainer.enable_progress_bar=false test=true predict=false data.scaling_config=src/data/scaling/scale_2020_JFM_hourly.yaml data.node_idx_csv=src/data/node_selection/node_idx_closest_center_10000.csv data.edge_weight_scale=10000

#uv run src/train.py --config-name=train_gcn_lstm data.batch_size=256 data.max_links=10000000 data.max_nodes=1000 data.max_radius=200000. model.optimizer.lr=0.0007 experiment=gcn_1h callbacks.early_stopping.patience=3 model.scheduler.patience=2 "data.train_days=[1, 90]" trainer.enable_progress_bar=false test=false predict=false data.scaling_config=src/data/scaling/scale_2020_JFM_hourly.yaml data.node_idx_csv=src/data/node_selection/node_idx_closest_center_10000.csv
#uv run src/train.py --config-name=train_gcn_lstm data.seq_len=3 data.batch_size=256 data.max_links=10000000 data.max_nodes=1000 data.max_radius=200000. model.optimizer.lr=0.0007 experiment=gcn_1h callbacks.early_stopping.patience=3 model.scheduler.patience=2 "data.train_days=[1, 90]" trainer.enable_progress_bar=false test=false predict=false data.scaling_config=src/data/scaling/scale_2020_JFM_hourly.yaml data.node_idx_csv=src/data/node_selection/node_idx_closest_center_10000.csv

#uv run src/train.py --config-name=train_gcn experiment=gcn_1h data.batch_size=256 test=false predict=false  "data.train_days=[1, 90]" data.node_idx_csv=src/data/node_selection/node_idx_closest_center_10000.csv model.focal_gamma=1.0

#uv run src/train.py --config-name=train_gcn_lstm_center_vn data.batch_size=256 data.node_idx_csv=src/data/node_selection/node_idx_closest_center_10000.csv  data.train_days="[1, 90]" experiment=gcn_1h data.seq_len=3 model.focal_gamma=1.0 test=false predict=false

#uv run src/train.py --config-name=train_gcn_lstm_center_vn data.edge_weight_scale=3000.0 model.backbone.dropout=0.7 model.backbone.hidden_dim=512 model.backbone.lstm_hidden_dim=128 model.backbone.num_gcn_layers=4 model.backbone.num_lstm_layers=1 model.focal_gamma=1.0 callbacks.early_stopping.patience=5 model.scheduler.patience=2 "data.train_days=[1, 90]" trainer.enable_progress_bar=false test=false predict=false data.scaling_config=src/data/scaling/scale_2020_JFM_hourly.yaml data.node_idx_csv=/work/ka1176/caroline/gitlab/wave-wind-model-emulator/src/data/node_selection/node_idx_closest_center_10000.csv data.seq_len=3 data.batch_size=256


## --- experiment schedule May 2026 ---

#uv run src/train.py --config-name=train_gcn_lstm_center_vn experiment=gcn_1h data.edge_weight_scale=7500.0 model.backbone.dropout=0.25 model.backbone.hidden_dim=512 model.backbone.lstm_hidden_dim=64 model.backbone.num_gcn_layers=4 model.backbone.num_lstm_layers=3 model.focal_gamma=0.35 callbacks.early_stopping.patience=5 model.scheduler.patience=2 "data.train_days=[1, 90]" trainer.enable_progress_bar=false test=false predict=false data.scaling_config=src/data/scaling/scale_2020_JFM_hourly.yaml data.node_idx_csv=/work/ka1176/caroline/gitlab/wave-wind-model-emulator/src/data/node_selection/node_idx_closest_center_10000.csv data.seq_len=48 data.batch_size=256

#uv run src/train.py --config-name=train experiment=gcn_1h  test=true predict=false

#uv run src/train.py --config-name=train_lstm experiment=gcn_1h model.backbone.dropout=0.1 model.backbone.hidden_dim=256 model.backbone.num_layers=4 model.optimizer.lr=0.0002 data.preload_data=true test=true predict=false data.seq_len=3

#uv run src/train.py --config-name=train_lstm data.sample_weight_power=8.86 model.backbone.dropout=0.25 model.backbone.hidden_dim=1024 model.backbone.num_layers=1 experiment=lstm_1h callbacks.early_stopping.patience=3 model.scheduler.patience=2 data.preload_data=true "data.train_days=[1, 90]" trainer.enable_progress_bar=false data.scaling_config=src/data/scaling/scale_2020_JFM_hourly.yaml data.node_idx_csv=/work/ka1176/caroline/gitlab/wave-wind-model-emulator/src/data/node_selection/node_idx_closest_center_10000.csv test=true predict=true data.seq_len=3 model.backbone.bidirectional=false data.batch_size=256 &
#uv run src/train.py --config-name=train_lstm data.sample_weight_power=8.86 model.backbone.dropout=0.25 model.backbone.hidden_dim=1024 model.backbone.num_layers=1 experiment=lstm_1h callbacks.early_stopping.patience=3 model.scheduler.patience=2 data.preload_data=true "data.train_days=[1, 90]" trainer.enable_progress_bar=false data.scaling_config=src/data/scaling/scale_2020_JFM_hourly.yaml data.node_idx_csv=/work/ka1176/caroline/gitlab/wave-wind-model-emulator/src/data/node_selection/node_idx_closest_center_10000.csv test=true predict=true data.seq_len=48 model.backbone.bidirectional=false data.batch_size=256

#uv run src/train.py --config-name=train_gcn_lstm_center_vn experiment=gcn_1h data.edge_weight_scale=7500.0 model.backbone.dropout=0.25 model.backbone.hidden_dim=512 model.backbone.lstm_hidden_dim=64 model.backbone.num_gcn_layers=4 model.backbone.num_lstm_layers=3 model.focal_gamma=0.35 callbacks.early_stopping.patience=5 model.scheduler.patience=2 "data.train_days=[1, 90]" trainer.enable_progress_bar=false test=false predict=false data.scaling_config=src/data/scaling/scale_2020_JFM_hourly.yaml data.node_idx_csv=/work/ka1176/caroline/gitlab/wave-wind-model-emulator/src/data/node_selection/node_idx_closest_center_10000.csv data.seq_len=3 data.batch_size=256

#uv run src/train.py --config-name=train_gcn_epd data.edge_weight_scale=7500.0 model.backbone.decoder_layers=7 model.backbone.dropout=0.03 model.backbone.encoder_layers=4 model.backbone.hidden_dim=64 model.backbone.num_mp_layers=7 model.backbone.use_residual=True experiment=gcn_1h test=false predict=false model.focal_gamma=0.35 data.train_days="[1,90]" data.scaling_config="src/data/scaling/scale_2020_JFM_hourly.yaml" data.node_idx_csv="src/data/node_selection/node_idx_closest_center_10000.csv" data.batch_size=256
#uv run src/train.py --config-name=train_gat_epd data.edge_weight_scale=7500.0 experiment=gcn_1h test=false predict=false model.focal_gamma=0.35 data.train_days="[1,90]" data.scaling_config="src/data/scaling/scale_2020_JFM_hourly.yaml" data.node_idx_csv="src/data/node_selection/node_idx_closest_center_10000.csv" data.batch_size=256

#uv run src/train.py --config-name=train_gcn_lstm_center_vn experiment=gcn_1h data.edge_weight_scale=7500.0 model.backbone.dropout=0.25 model.backbone.hidden_dim=512 model.backbone.lstm_hidden_dim=64 model.backbone.num_gcn_layers=4 model.backbone.num_lstm_layers=3 model.focal_gamma=0.35 callbacks.early_stopping.patience=5 model.scheduler.patience=2 "data.train_days=[1, 90]" trainer.enable_progress_bar=false test=false predict=false data.scaling_config=src/data/scaling/scale_2020_JFM_hourly.yaml data.node_idx_csv=/work/ka1176/caroline/gitlab/wave-wind-model-emulator/src/data/node_selection/node_idx_closest_center_10000.csv data.seq_len=3 data.batch_size=256

#uv run src/train.py --config-name=train_gat_epd data.edge_weight_scale=12500 model.backbone.attn_dropout=0.20 model.backbone.decoder_layers=3 model.backbone.dropout=0.05 model.backbone.encoder_layers=2 model.backbone.hidden_dim=256 model.backbone.num_mp_layers=9 model.backbone.use_edge_features=False model.backbone.use_residual=True model.focal_gamma=0.8 experiment=gcn_1h callbacks.early_stopping.patience=3 model.scheduler.patience=2 "data.train_days=[1, 90]" trainer.enable_progress_bar=false test=false predict=false data.scaling_config=src/data/scaling/scale_2020_JFM_hourly.yaml data.node_idx_csv=/work/ka1176/caroline/gitlab/wave-wind-model-emulator/src/data/node_selection/node_idx_closest_center_10000.csv data.batch_size=256 model.optimizer.lr=0.0007 trainer.devices=2

#uv run src/train.py --config-name=train_gcn_lstm_center_vn experiment=gcn_1h model.backbone._target_=src.model.backbones.CrossAttnVirtualNodeGCN.CrossAttnVirtualNodeGCN data.edge_weight_scale=7500.0 model.backbone.dropout=0.25 model.backbone.hidden_dim=512 model.backbone.lstm_hidden_dim=64 model.backbone.num_gcn_layers=4 model.backbone.num_lstm_layers=3 model.focal_gamma=0.35 callbacks.early_stopping.patience=5 model.scheduler.patience=2 "data.train_days=[1, 90]" trainer.enable_progress_bar=false test=false predict=false data.scaling_config=src/data/scaling/scale_2020_JFM_hourly.yaml data.node_idx_csv=/work/ka1176/caroline/gitlab/wave-wind-model-emulator/src/data/node_selection/node_idx_closest_center_10000.csv data.seq_len=3 data.batch_size=256
#uv run src/train.py --config-name=train_gcn_lstm_center_vn experiment=gcn_1h model.backbone._target_=src.model.backbones.CrossAttnVirtualNodeGCN.CrossAttnVirtualNodeGCN data.edge_weight_scale=7500.0 model.backbone.dropout=0.25 model.backbone.hidden_dim=512 model.backbone.lstm_hidden_dim=64 model.backbone.num_gcn_layers=4 model.backbone.num_lstm_layers=3 model.focal_gamma=0.35 callbacks.early_stopping.patience=5 model.scheduler.patience=2 "data.train_days=[1, 90]" trainer.enable_progress_bar=false test=false predict=false data.scaling_config=src/data/scaling/scale_2020_JFM_hourly.yaml data.node_idx_csv=/work/ka1176/caroline/gitlab/wave-wind-model-emulator/src/data/node_selection/node_idx_closest_center_10000.csv data.seq_len=48 data.batch_size=256 'ckpt_path="./logs/train_gcn_lstm_center_vn/runs/2026-05-19_14-38-52/wwm-emulator-hourly/144os7dr/checkpoints/last.ckpt"' logger.wandb.id="144os7dr"

#uv run src/train.py --config-name=train_dual_path_gat_vn experiment=gcn_1h trainer.devices=1 logger.wandb.id="ijomh6ep" 'ckpt_path="logs/train_dual_path_gat_vn/runs/2026-06-04_11-12-20/wwm-emulator-hourly/ijomh6ep/checkpoints/last.ckpt"'
#uv run src/train.py --config-name=train_multihead_gat experiment=gcn_1h trainer.devices=2


#uv run src/train.py --config-name=train_gat_epd experiment=gcn_1h data.sample_weight_strategy="z_score" data.sample_weight_power=20.0 trainer.devices=2 data.batch_size=256

#uv run src/train.py experiment=mwp_1h_lstm test=false trainer.devices=4 logger.wandb.project="wwm-emulator-hourly" data.sample_weight_power=0.0
#uv run src/train.py experiment=mwp_1h_lstm test=false trainer.devices=4 logger.wandb.project="wwm-emulator-hourly" data.sample_weight_power=1.0
#uv run src/train.py experiment=mwp_1h_lstm test=false trainer.devices=4 logger.wandb.project="wwm-emulator-hourly" data.sample_weight_power=5.0
#uv run src/train.py experiment=mwp_1h_lstm test=false trainer.devices=4 logger.wandb.project="wwm-emulator-hourly" data.sample_weight_power=10.0
#uv run src/train.py experiment=mwp_1h_lstm test=false trainer.devices=4 logger.wandb.project="wwm-emulator-hourly" data.sample_weight_power=20.0
#uv run src/train.py experiment=mwp_1h_lstm test=false trainer.devices=4 logger.wandb.project="wwm-emulator-hourly" data.sample_weight_power=50.0
#uv run src/train.py experiment=mwp_1h_lstm test=false trainer.devices=4 logger.wandb.project="wwm-emulator-hourly" data.sample_weight_power=100.0

uv run src/train.py experiment=swh_mwp_1h_lstm test=false trainer.devices=4 logger.wandb.project="wwm-emulator-hourly"

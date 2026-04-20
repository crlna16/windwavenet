#!/bin/bash
#SBATCH -A ka1176
#SBATCH -p vader
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH -o scripts/slurm/slurm-%j.out

export HYDRA_FULL_ERROR=1

#uv run src/predict.py logger=null tags="[default]"

#uv run src/predict.py logger=null experiment=best_sweep_na1becz6 'ckpt_path="/work/ka1176/caroline/gitlab/wave-wind-model-emulator/logs/train/best_sweep_na1becz6/runs/2025-12-08_09-49-35/simple-wwm-emulator/rvmycqfs/checkpoints/epoch=1-step=120434.ckpt"'
#uv run src/predict.py logger=null experiment=best_sweep_ju8o025p 'ckpt_path="/work/ka1176/caroline/gitlab/wave-wind-model-emulator/logs/train/densenet_8_features/runs/2025-12-09_09-19-47/simple-wwm-emulator/1b4t6477/checkpoints/epoch=2-step=134226.ckpt"'
#uv run src/predict.py logger=null experiment=densenet_SWH model.backbone.num_hidden_layers=2 model.backbone.layer_scaling="halving" model.backbone.first_layer_neurons=32 'ckpt_path="/work/ka1176/caroline/gitlab/wave-wind-model-emulator/logs/train/densenet_SWH/runs/2025-12-09_14-05-19/simple-wwm-emulator/x1btoy5i/checkpoints/epoch=4-step=252705.ckpt"'
#uv run src/train_ml.py logger=null experiment=xgboost_8_features data.sample_weight_power=98.34 model.model.colsample_bylevel=0.526 model.model.colsample_bytree=0.905 model.model.eta=0.329 model.model.gamma=4.48 model.model.lambda=1.928 model.model.learning_rate=0.442 model.model.max_depth=9 model.model.n_estimators=112 model.model.subsample=0.972 predict=true
#uv run src/predict.py logger=null experiment=densenet_8_features data.sample_weight_strategy="z_score" data.batch_size=2787 data.sample_weight_power=84.77 model.backbone.first_layer_neurons=512 model.backbone.layer_scaling="halving" model.backbone.num_hidden_layers=2 'ckpt_path="/work/ka1176/caroline/gitlab/wave-wind-model-emulator/logs/train/densenet_8_features/runs/2026-01-16_20-18-40/simple-wwm-emulator/k9lhx9xr/checkpoints/epoch=0-step=96378.ckpt"'

#uv run src/predict.py model.backbone.dropout_rate=0.2751473580946915 model.backbone.first_layer_neurons=512 model.backbone.layer_scaling="halving" model.backbone.num_hidden_layers=2 model.optimizer.lr=0.0004014181296452297 trainer.val_check_interval=868 experiment=densenet_1h data.scaling_config=src/data/scaling/scale_2020_JFM_hourly.yaml data.targets=["sigWaveHeight"] 'ckpt_path="/work/ka1176/caroline/gitlab/wave-wind-model-emulator/logs/train/densenet_1h/runs/2026-02-14_07-57-28/wwm-emulator-hourly/kejwqfeq/checkpoints/epoch=0-step=9548.ckpt"'
#uv run src/predict.py  model.backbone.first_layer_neurons=1024 model.backbone.layer_scaling="constant" model.backbone.num_hidden_layers=1 experiment=densenet_1h data.scaling_config=src/data/scaling/scale_2020_JFM_hourly.yaml 'ckpt_path="/work/ka1176/caroline/gitlab/wave-wind-model-emulator/logs/train/densenet_1h/runs/2026-02-15_02-15-11/wwm-emulator-hourly/ezrhbaya/checkpoints/epoch=0-step=20040.ckpt"'

#uv run src/predict.py --config-name=predict_lstm data.seq_len=48 model.backbone.bidirectional=False model.backbone.dropout=0.13762518479545507 model.backbone.hidden_dim=128 model.backbone.num_layers=4 model.optimizer.lr=0.00011203848060182346 experiment=lstm_1h data.preload_data=true "data.train_days=[1, 90]" trainer.enable_progress_bar=true data.scaling_config=src/data/scaling/scale_2020_JFM_hourly.yaml data.node_idx_csv=/work/ka1176/caroline/gitlab/wave-wind-model-emulator/src/data/node_selection/node_idx_closest_center_10000.csv 'ckpt_path="/work/ka1176/caroline/gitlab/wave-wind-model-emulator/logs/train_lstm/lstm_1h/runs/2026-02-24_04-36-31/wwm-emulator-hourly/oe3dqt9e/checkpoints/epoch=9-step=50839.ckpt"'

#uv run src/predict.py --config-name=predict_lstm model.backbone.bidirectional=False model.backbone.dropout=0.029216092168033957 model.backbone.hidden_dim=256 model.backbone.num_layers=4  experiment=lstm_1h data.scaling_config=src/data/scaling/scale_2020_JFM_hourly.yaml data.node_idx_csv=/work/ka1176/caroline/gitlab/wave-wind-model-emulator/src/data/node_selection/node_idx_closest_center_10000.csv data.seq_len=6 'ckpt_path="/work/ka1176/caroline/gitlab/wave-wind-model-emulator/logs/train_lstm/lstm_1h/runs/2026-03-18_22-37-25/wwm-emulator-hourly/anslcihw/checkpoints/epoch=11-step=62329.ckpt"'

#uv run src/predict.py --config-name=predict_lstm experiment=lstm_1h model.backbone.bidirectional=True model.backbone.hidden_dim=256 model.backbone.num_layers=4 data.scaling_config=src/data/scaling/scale_2020_JFM_hourly.yaml data.seq_len=3 'ckpt_path="/work/ka1176/caroline/gitlab/wave-wind-model-emulator/logs/train_lstm/lstm_1h/runs/2026-03-19_23-48-05/wwm-emulator-hourly/y5gdkgve/checkpoints/epoch=15-step=84098.ckpt"'

#uv run src/predict.py --config-name=predict_lstm model.backbone.bidirectional=True model.backbone.dropout=0.1 model.backbone.hidden_dim=256 model.backbone.num_layers=4 experiment=lstm_1h trainer.enable_progress_bar=false data.scaling_config=src/data/scaling/scale_2020_JFM_hourly.yaml data.seq_len=3 'ckpt_path="/work/ka1176/caroline/gitlab/wave-wind-model-emulator/logs/train_lstm/lstm_1h/runs/2026-03-25_12-34-29/wwm-emulator-hourly/3gsfe50s/checkpoints/epoch=35-step=190476.ckpt"'

#uv run src/predict_gcn.py --config-name=predict_gcn_epd data.batch_size=256 data.max_links=10000000 data.max_nodes=1000 data.max_radius=200000. model.backbone.decoder_layers=2 model.backbone.dropout=0.1 model.backbone.encoder_layers=2 model.backbone.hidden_dim=32 model.backbone.num_mp_layers=10 model.backbone.use_residual=True experiment=gcn_1h data.scaling_config=src/data/scaling/scale_2020_JFM_hourly.yaml  data.edge_weight_scale=10000 'ckpt_path="logs/train_gcn_epd/gcn_1h/runs/2026-04-13_12-44-19/wwm-emulator-hourly/ky6nafyr/checkpoints/epoch=6-step=148288.ckpt"' logger.wandb.id="ky6nafyr" test_metrics=true

uv run src/predict_gcn.py --config-name=predict_gcn_lstm data.seq_len=3 data.batch_size=256 data.max_links=10000000 data.max_nodes=1000 data.max_radius=200000. experiment=gcn_1h data.scaling_config=src/data/scaling/scale_2020_JFM_hourly.yaml test_metrics=true logger.wandb.id="gva36ir9" 'ckpt_path="logs/train_gcn_lstm/gcn_1h/runs/2026-04-14_11-12-28/wwm-emulator-hourly/gva36ir9/checkpoints/epoch=9-step=211640.ckpt"'

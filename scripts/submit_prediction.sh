#!/bin/bash
#SBATCH -A ka1176
#SBATCH -p vader
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH -o scripts/slurm/slurm-%j.out

#uv run src/predict.py logger=null tags="[default]"

#uv run src/predict.py logger=null experiment=best_sweep_na1becz6 'ckpt_path="/work/ka1176/caroline/gitlab/wave-wind-model-emulator/logs/train/best_sweep_na1becz6/runs/2025-12-08_09-49-35/simple-wwm-emulator/rvmycqfs/checkpoints/epoch=1-step=120434.ckpt"'
#uv run src/predict.py logger=null experiment=best_sweep_ju8o025p 'ckpt_path="/work/ka1176/caroline/gitlab/wave-wind-model-emulator/logs/train/densenet_8_features/runs/2025-12-09_09-19-47/simple-wwm-emulator/1b4t6477/checkpoints/epoch=2-step=134226.ckpt"'
#uv run src/predict.py logger=null experiment=densenet_SWH model.backbone.num_hidden_layers=2 model.backbone.layer_scaling="halving" model.backbone.first_layer_neurons=32 'ckpt_path="/work/ka1176/caroline/gitlab/wave-wind-model-emulator/logs/train/densenet_SWH/runs/2025-12-09_14-05-19/simple-wwm-emulator/x1btoy5i/checkpoints/epoch=4-step=252705.ckpt"'
#uv run src/train_ml.py logger=null experiment=xgboost_8_features data.sample_weight_power=98.34 model.model.colsample_bylevel=0.526 model.model.colsample_bytree=0.905 model.model.eta=0.329 model.model.gamma=4.48 model.model.lambda=1.928 model.model.learning_rate=0.442 model.model.max_depth=9 model.model.n_estimators=112 model.model.subsample=0.972 predict=true
uv run src/predict.py logger=null experiment=densenet_8_features data.sample_weight_strategy="z_score" data.batch_size=2787 data.sample_weight_power=84.77 model.backbone.first_layer_neurons=512 model.backbone.layer_scaling="halving" model.backbone.num_hidden_layers=2 'ckpt_path="/work/ka1176/caroline/gitlab/wave-wind-model-emulator/logs/train/densenet_8_features/runs/2026-01-16_20-18-40/simple-wwm-emulator/k9lhx9xr/checkpoints/epoch=0-step=96378.ckpt"'

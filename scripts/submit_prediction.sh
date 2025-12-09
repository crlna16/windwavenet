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
uv run src/predict.py logger=null experiment=densenet_SWH model.backbone.num_hidden_layers=2 model.backbone.layer_scaling="halving" model.backbone.first_layer_neurons=32 'ckpt_path="/work/ka1176/caroline/gitlab/wave-wind-model-emulator/logs/train/densenet_SWH/runs/2025-12-09_14-05-19/simple-wwm-emulator/x1btoy5i/checkpoints/epoch=4-step=252705.ckpt"'

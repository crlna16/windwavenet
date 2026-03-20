#!/bin/bash
#SBATCH -A ka1176
#SBATCH -p vader
#SBATCH --array=0-1
#SBATCH --mem=200G
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00
#SBATCH -o scripts/slurm/slurm-%j.out

#wandb agent wwm-emulator/simple-wwm-emulator/w5i9gsye
#wandb agent wwm-emulator/simple-wwm-emulator/pneifth9
#wandb agent wwm-emulator/simple-wwm-emulator/na1becz6
#wandb agent wwm-emulator/simple-wwm-emulator/ju8o025p
#wandb agent wwm-emulator/simple-wwm-emulator/ql1ymdn4
#wandb agent wwm-emulator/simple-wwm-emulator/fozrigjd
#wandb agent wwm-emulator/simple-wwm-emulator/p5pk1arc
#wandb agent wwm-emulator/simple-wwm-emulator/l9lwmyqf # xgboost
#wandb agent wwm-emulator/simple-wwm-emulator/o2yosk20 # densenet
#wandb agent wwm-emulator/simple-wwm-emulator/btsgf6ro # xgboost sigWaveHeight
#wandb agent wwm-emulator/simple-wwm-emulator/ms3xo7a2 # xgboost subsample
#wandb agent wwm-emulator/simple-wwm-emulator/xpzhovm8
#wandb agent wwm-emulator/wwm-emulator-hourly/r31hnrh0
uv run wandb agent wwm-emulator/wwm-emulator-hourly/7lyrjwwo

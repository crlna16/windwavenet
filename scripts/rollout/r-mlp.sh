#!/bin/bash
#SBATCH -A ka1176
#SBATCH -p gpu
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH -o scripts/slurm/slurm-%A_%a.out

## Run a rollout
# Needs one run per forecast start time
# Staggered with seeds
# Seed 0, 1, 2: day 0, 00:00
# Seed 0, 1, 2: day 1, 08:00
# Seed 0, 1, 2: day 2, 16:00
# Seed 0, 1, 2: day 3, 00:00
# ...
# Two blocks for summer [183,213] and winter [336,366]

declare -A CKPTS=(
    [172426]="logs/train/swh_1h_densenet/runs/2026-09-10_14-22-52/wwm-emulator-hourly-final/sytiiae9/checkpoints/epoch=58-step=214229.ckpt"
    [211100]="logs/train/swh_1h_densenet/runs/2026-10-18_21-14-45/wwm-emulator-hourly-final/o3iw1q9t/checkpoints/epoch=50-step=185130.ckpt"
    [110814]="logs/train/swh_1h_densenet/runs/2026-10-19_11-15-38/wwm-emulator-hourly-final/vrd9q0t1/checkpoints/epoch=44-step=163350.ckpt"
)

# Loop seeds
for SEED in "${!CKPTS[@]}"; do
    # summer
    uv run python src/rollout.py experiment=swh_1h_mlp data.test_days='[183,213]' rollout.offset_hours=0 paths.output_dir="./logs/rollout2/swh-mlp-seed-${SEED}" logger=null "ckpt_path='${CKPTS[$SEED]}'"
    uv run python src/rollout.py experiment=swh_1h_mlp data.test_days='[184,213]' rollout.offset_hours=8 paths.output_dir="./logs/rollout2/swh-mlp-seed-${SEED}" logger=null "ckpt_path='${CKPTS[$SEED]}'"
    uv run python src/rollout.py experiment=swh_1h_mlp data.test_days='[185,213]' rollout.offset_hours=16 paths.output_dir="./logs/rollout2/swh-mlp-seed-${SEED}" logger=null "ckpt_path='${CKPTS[$SEED]}'"
    uv run python src/rollout.py experiment=swh_1h_mlp data.test_days='[186,213]' rollout.offset_hours=0 paths.output_dir="./logs/rollout2/swh-mlp-seed-${SEED}" logger=null "ckpt_path='${CKPTS[$SEED]}'"
    uv run python src/rollout.py experiment=swh_1h_mlp data.test_days='[187,213]' rollout.offset_hours=8 paths.output_dir="./logs/rollout2/swh-mlp-seed-${SEED}" logger=null "ckpt_path='${CKPTS[$SEED]}'"
    uv run python src/rollout.py experiment=swh_1h_mlp data.test_days='[188,213]' rollout.offset_hours=16 paths.output_dir="./logs/rollout2/swh-mlp-seed-${SEED}" logger=null "ckpt_path='${CKPTS[$SEED]}'"
    uv run python src/rollout.py experiment=swh_1h_mlp data.test_days='[189,213]' rollout.offset_hours=0 paths.output_dir="./logs/rollout2/swh-mlp-seed-${SEED}" logger=null "ckpt_path='${CKPTS[$SEED]}'"
    uv run python src/rollout.py experiment=swh_1h_mlp data.test_days='[190,213]' rollout.offset_hours=8 paths.output_dir="./logs/rollout2/swh-mlp-seed-${SEED}" logger=null "ckpt_path='${CKPTS[$SEED]}'"
    uv run python src/rollout.py experiment=swh_1h_mlp data.test_days='[191,213]' rollout.offset_hours=16 paths.output_dir="./logs/rollout2/swh-mlp-seed-${SEED}" logger=null "ckpt_path='${CKPTS[$SEED]}'"
    uv run python src/rollout.py experiment=swh_1h_mlp data.test_days='[192,213]' rollout.offset_hours=0 paths.output_dir="./logs/rollout2/swh-mlp-seed-${SEED}" logger=null "ckpt_path='${CKPTS[$SEED]}'"
    uv run python src/rollout.py experiment=swh_1h_mlp data.test_days='[193,213]' rollout.offset_hours=8 paths.output_dir="./logs/rollout2/swh-mlp-seed-${SEED}" logger=null "ckpt_path='${CKPTS[$SEED]}'"
    uv run python src/rollout.py experiment=swh_1h_mlp data.test_days='[194,213]' rollout.offset_hours=16 paths.output_dir="./logs/rollout2/swh-mlp-seed-${SEED}" logger=null "ckpt_path='${CKPTS[$SEED]}'"
    uv run python src/rollout.py experiment=swh_1h_mlp data.test_days='[195,213]' rollout.offset_hours=0 paths.output_dir="./logs/rollout2/swh-mlp-seed-${SEED}" logger=null "ckpt_path='${CKPTS[$SEED]}'"
    uv run python src/rollout.py experiment=swh_1h_mlp data.test_days='[196,213]' rollout.offset_hours=8 paths.output_dir="./logs/rollout2/swh-mlp-seed-${SEED}" logger=null "ckpt_path='${CKPTS[$SEED]}'"
    uv run python src/rollout.py experiment=swh_1h_mlp data.test_days='[197,213]' rollout.offset_hours=16 paths.output_dir="./logs/rollout2/swh-mlp-seed-${SEED}" logger=null "ckpt_path='${CKPTS[$SEED]}'"
    uv run python src/rollout.py experiment=swh_1h_mlp data.test_days='[198,213]' rollout.offset_hours=0 paths.output_dir="./logs/rollout2/swh-mlp-seed-${SEED}" logger=null "ckpt_path='${CKPTS[$SEED]}'"
    uv run python src/rollout.py experiment=swh_1h_mlp data.test_days='[199,213]' rollout.offset_hours=8 paths.output_dir="./logs/rollout2/swh-mlp-seed-${SEED}" logger=null "ckpt_path='${CKPTS[$SEED]}'"
    uv run python src/rollout.py experiment=swh_1h_mlp data.test_days='[200,213]' rollout.offset_hours=16 paths.output_dir="./logs/rollout2/swh-mlp-seed-${SEED}" logger=null "ckpt_path='${CKPTS[$SEED]}'"
    uv run python src/rollout.py experiment=swh_1h_mlp data.test_days='[201,213]' rollout.offset_hours=0 paths.output_dir="./logs/rollout2/swh-mlp-seed-${SEED}" logger=null "ckpt_path='${CKPTS[$SEED]}'"
    uv run python src/rollout.py experiment=swh_1h_mlp data.test_days='[202,213]' rollout.offset_hours=8 paths.output_dir="./logs/rollout2/swh-mlp-seed-${SEED}" logger=null "ckpt_path='${CKPTS[$SEED]}'"
    uv run python src/rollout.py experiment=swh_1h_mlp data.test_days='[203,213]' rollout.offset_hours=16 paths.output_dir="./logs/rollout2/swh-mlp-seed-${SEED}" logger=null "ckpt_path='${CKPTS[$SEED]}'"
    uv run python src/rollout.py experiment=swh_1h_mlp data.test_days='[204,213]' rollout.offset_hours=0 paths.output_dir="./logs/rollout2/swh-mlp-seed-${SEED}" logger=null "ckpt_path='${CKPTS[$SEED]}'"
    # summer
    uv run python src/rollout.py experiment=swh_1h_mlp data.test_days='[336,366]' rollout.offset_hours=0 paths.output_dir="./logs/rollout2/swh-mlp-seed-${SEED}" logger=null "ckpt_path='${CKPTS[$SEED]}'"
    uv run python src/rollout.py experiment=swh_1h_mlp data.test_days='[337,366]' rollout.offset_hours=8 paths.output_dir="./logs/rollout2/swh-mlp-seed-${SEED}" logger=null "ckpt_path='${CKPTS[$SEED]}'"
    uv run python src/rollout.py experiment=swh_1h_mlp data.test_days='[338,366]' rollout.offset_hours=16 paths.output_dir="./logs/rollout2/swh-mlp-seed-${SEED}" logger=null "ckpt_path='${CKPTS[$SEED]}'"
    uv run python src/rollout.py experiment=swh_1h_mlp data.test_days='[339,366]' rollout.offset_hours=0 paths.output_dir="./logs/rollout2/swh-mlp-seed-${SEED}" logger=null "ckpt_path='${CKPTS[$SEED]}'"
    uv run python src/rollout.py experiment=swh_1h_mlp data.test_days='[340,366]' rollout.offset_hours=8 paths.output_dir="./logs/rollout2/swh-mlp-seed-${SEED}" logger=null "ckpt_path='${CKPTS[$SEED]}'"
    uv run python src/rollout.py experiment=swh_1h_mlp data.test_days='[341,366]' rollout.offset_hours=16 paths.output_dir="./logs/rollout2/swh-mlp-seed-${SEED}" logger=null "ckpt_path='${CKPTS[$SEED]}'"
    uv run python src/rollout.py experiment=swh_1h_mlp data.test_days='[342,366]' rollout.offset_hours=0 paths.output_dir="./logs/rollout2/swh-mlp-seed-${SEED}" logger=null "ckpt_path='${CKPTS[$SEED]}'"
    uv run python src/rollout.py experiment=swh_1h_mlp data.test_days='[343,366]' rollout.offset_hours=8 paths.output_dir="./logs/rollout2/swh-mlp-seed-${SEED}" logger=null "ckpt_path='${CKPTS[$SEED]}'"
    uv run python src/rollout.py experiment=swh_1h_mlp data.test_days='[344,366]' rollout.offset_hours=16 paths.output_dir="./logs/rollout2/swh-mlp-seed-${SEED}" logger=null "ckpt_path='${CKPTS[$SEED]}'"
    uv run python src/rollout.py experiment=swh_1h_mlp data.test_days='[345,366]' rollout.offset_hours=0 paths.output_dir="./logs/rollout2/swh-mlp-seed-${SEED}" logger=null "ckpt_path='${CKPTS[$SEED]}'"
    uv run python src/rollout.py experiment=swh_1h_mlp data.test_days='[346,366]' rollout.offset_hours=8 paths.output_dir="./logs/rollout2/swh-mlp-seed-${SEED}" logger=null "ckpt_path='${CKPTS[$SEED]}'"
    uv run python src/rollout.py experiment=swh_1h_mlp data.test_days='[347,366]' rollout.offset_hours=16 paths.output_dir="./logs/rollout2/swh-mlp-seed-${SEED}" logger=null "ckpt_path='${CKPTS[$SEED]}'"
    uv run python src/rollout.py experiment=swh_1h_mlp data.test_days='[348,366]' rollout.offset_hours=0 paths.output_dir="./logs/rollout2/swh-mlp-seed-${SEED}" logger=null "ckpt_path='${CKPTS[$SEED]}'"
    uv run python src/rollout.py experiment=swh_1h_mlp data.test_days='[349,366]' rollout.offset_hours=8 paths.output_dir="./logs/rollout2/swh-mlp-seed-${SEED}" logger=null "ckpt_path='${CKPTS[$SEED]}'"
    uv run python src/rollout.py experiment=swh_1h_mlp data.test_days='[350,366]' rollout.offset_hours=16 paths.output_dir="./logs/rollout2/swh-mlp-seed-${SEED}" logger=null "ckpt_path='${CKPTS[$SEED]}'"
    uv run python src/rollout.py experiment=swh_1h_mlp data.test_days='[351,366]' rollout.offset_hours=0 paths.output_dir="./logs/rollout2/swh-mlp-seed-${SEED}" logger=null "ckpt_path='${CKPTS[$SEED]}'"
    uv run python src/rollout.py experiment=swh_1h_mlp data.test_days='[352,366]' rollout.offset_hours=8 paths.output_dir="./logs/rollout2/swh-mlp-seed-${SEED}" logger=null "ckpt_path='${CKPTS[$SEED]}'"
    uv run python src/rollout.py experiment=swh_1h_mlp data.test_days='[353,366]' rollout.offset_hours=16 paths.output_dir="./logs/rollout2/swh-mlp-seed-${SEED}" logger=null "ckpt_path='${CKPTS[$SEED]}'"
    uv run python src/rollout.py experiment=swh_1h_mlp data.test_days='[354,366]' rollout.offset_hours=0 paths.output_dir="./logs/rollout2/swh-mlp-seed-${SEED}" logger=null "ckpt_path='${CKPTS[$SEED]}'"
    uv run python src/rollout.py experiment=swh_1h_mlp data.test_days='[355,366]' rollout.offset_hours=8 paths.output_dir="./logs/rollout2/swh-mlp-seed-${SEED}" logger=null "ckpt_path='${CKPTS[$SEED]}'"
    uv run python src/rollout.py experiment=swh_1h_mlp data.test_days='[356,366]' rollout.offset_hours=16 paths.output_dir="./logs/rollout2/swh-mlp-seed-${SEED}" logger=null "ckpt_path='${CKPTS[$SEED]}'"
    uv run python src/rollout.py experiment=swh_1h_mlp data.test_days='[357,366]' rollout.offset_hours=0 paths.output_dir="./logs/rollout2/swh-mlp-seed-${SEED}" logger=null "ckpt_path='${CKPTS[$SEED]}'"
done

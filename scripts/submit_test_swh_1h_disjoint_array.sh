#!/bin/bash
#SBATCH -A ka1176
#SBATCH -p vader
#SBATCH --mem=200G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH -o scripts/slurm/slurm-%A_%a.out
#SBATCH --array=0-34

# how to call
# sbatch --array=15-18 scripts/submit_swh_1h_array.sh

COMMANDS=(
    # DenseNet (0)
    "uv run python src/predict.py experiment=swh_1h_mlp test_metrics=true paths.output_dir='/work/ka1176/caroline/gitlab/wave-wind-model-emulator/logs/predict_disjoint_final/swh-mlp' logger.wandb.id=9m79pxc0 'ckpt_path=\"logs/train/swh_1h_densenet/runs/2026-07-23_09-53-43/wwm-emulator-hourly-final/9m79pxc0/checkpoints/epoch=51-step=188812.ckpt\"'"
    # LSTM (1-5)
    "uv run python src/predict.py --config-name=predict_lstm experiment=swh_1h_lstm data.seq_len=3 paths.output_dir='/work/ka1176/caroline/gitlab/wave-wind-model-emulator/logs/predict_disjoint_final/swh-lstm-3' logger.wandb.id=pxvonspr 'ckpt_path=\"logs/train/swh_1h_lstm/runs/2026-07-22_05-59-48/wwm-emulator-hourly-final/pxvonspr/checkpoints/epoch=63-step=464384.ckpt\"'"
    "uv run python src/predict.py --config-name=predict_lstm experiment=swh_1h_lstm data.seq_len=6 paths.output_dir='/work/ka1176/caroline/gitlab/wave-wind-model-emulator/logs/predict_disjoint_final/swh-lstm-6' logger.wandb.id=6vxr1ldl 'ckpt_path=\"logs/train/swh_1h_lstm/runs/2026-07-23_16-24-50/wwm-emulator-hourly-final/6vxr1ldl/checkpoints/epoch=32-step=239184.ckpt\"'"
    "uv run python src/predict.py --config-name=predict_lstm experiment=swh_1h_lstm data.seq_len=12 paths.output_dir='/work/ka1176/caroline/gitlab/wave-wind-model-emulator/logs/predict_disjoint_final/swh-lstm-12' logger.wandb.id=nqsp08sf 'ckpt_path=\"logs/train/swh_1h_lstm/runs/2026-07-22_07-20-44/wwm-emulator-hourly-final/nqsp08sf/checkpoints/epoch=20-step=151914.ckpt\"'"
    "uv run python src/predict.py --config-name=predict_lstm experiment=swh_1h_lstm data.seq_len=24 paths.output_dir='/work/ka1176/caroline/gitlab/wave-wind-model-emulator/logs/predict_disjoint_final/swh-lstm-24' logger.wandb.id=vnzc7qo5 'ckpt_path=\"logs/train/swh_1h_lstm/runs/2026-07-22_20-22-54/wwm-emulator-hourly-final/vnzc7qo5/checkpoints/epoch=21-step=158488.ckpt\"'"
    "uv run python src/predict.py --config-name=predict_lstm experiment=swh_1h_lstm data.seq_len=48 paths.output_dir='/work/ka1176/caroline/gitlab/wave-wind-model-emulator/logs/predict_disjoint_final/swh-lstm-48' logger.wandb.id=69ybnp2j 'ckpt_path=\"logs/train/swh_1h_lstm/runs/2026-07-22_21-17-33/wwm-emulator-hourly-final/69ybnp2j/checkpoints/epoch=33-step=242896.ckpt\"'"
    # Multi-Head GAT (6-8)
    "uv run python src/predict_gcn.py experiment=swh_1h_multi-head-gat.yaml data.max_nodes=500 paths.output_dir='/work/ka1176/caroline/gitlab/wave-wind-model-emulator/logs/predict_disjoint_final/swh-gat-500' logger.wandb.id=6lvzw95a 'ckpt_path=\"logs/train/swh_1h_multi-head-gat/runs/2026-08-21_09-51-48/wwm-emulator-hourly-final/6lvzw95a/checkpoints/epoch=12-step=188773.ckpt\"'"
    "uv run python src/predict_gcn.py experiment=swh_1h_multi-head-gat.yaml data.max_nodes=250 paths.output_dir='/work/ka1176/caroline/gitlab/wave-wind-model-emulator/logs/predict_disjoint_final/swh-gat-250' logger.wandb.id=svpmuzlx 'ckpt_path=\"logs/train/swh_1h_multi-head-gat/runs/2026-08-21_09-54-27/wwm-emulator-hourly-final/svpmuzlx/checkpoints/epoch=2-step=43563.ckpt\"'"
    "uv run python src/predict_gcn.py experiment=swh_1h_multi-head-gat.yaml data.max_nodes=50 paths.output_dir='/work/ka1176/caroline/gitlab/wave-wind-model-emulator/logs/predict_disjoint_final/swh-gat-50' logger.wandb.id=zmmeh8yr 'ckpt_path=\"logs/train/swh_1h_multi-head-gat/runs/2026-08-21_09-55-30/wwm-emulator-hourly-final/zmmeh8yr/checkpoints/epoch=6-step=101647.ckpt\"'"
    # Multi-Head GAT VN (9-11)
    "uv run python src/predict_gcn.py experiment=swh_1h_multi-head-gat-vn.yaml data.max_nodes=500 paths.output_dir='/work/ka1176/caroline/gitlab/wave-wind-model-emulator/logs/predict_disjoint_final/swh-gat-vn-500' logger.wandb.id=n2j2gloz 'ckpt_path=\"logs/train/swh_1h_multi-head-gat-vn/runs/2026-08-21_09-57-12/wwm-emulator-hourly-final/n2j2gloz/checkpoints/epoch=9-step=145210.ckpt\"'"
    "uv run python src/predict_gcn.py experiment=swh_1h_multi-head-gat-vn.yaml data.max_nodes=250 paths.output_dir='/work/ka1176/caroline/gitlab/wave-wind-model-emulator/logs/predict_disjoint_final/swh-gat-vn-250' logger.wandb.id=l19h6pc1 'ckpt_path=\"logs/train/swh_1h_multi-head-gat-vn/runs/2026-08-21_09-58-28/wwm-emulator-hourly-final/l19h6pc1/checkpoints/epoch=8-step=130689.ckpt\"'"
    "uv run python src/predict_gcn.py experiment=swh_1h_multi-head-gat-vn.yaml data.max_nodes=50 paths.output_dir='/work/ka1176/caroline/gitlab/wave-wind-model-emulator/logs/predict_disjoint_final/swh-gat-vn-50' logger.wandb.id=2w4mhqs5 'ckpt_path=\"logs/train/swh_1h_multi-head-gat-vn/runs/2026-08-21_13-33-33/wwm-emulator-hourly-final/2w4mhqs5/checkpoints/epoch=28-step=421109.ckpt\"'"
    # Multi-Head GAT LSTM VN (12-17)
    "uv run python src/predict_gcn.py experiment=swh_1h_multi-head-gat-lstm-vn.yaml data.seq_len=6 data.max_nodes=500 paths.output_dir='/work/ka1176/caroline/gitlab/wave-wind-model-emulator/logs/predict_disjoint_final/swh-gat-lstm-vn-6-500' logger.wandb.id=7gg6p6wq 'ckpt_path=\"logs/train/swh_1h_multi-head-gat-lstm-vn/runs/2026-08-21_16-53-35/wwm-emulator-hourly-final/7gg6p6wq/checkpoints/epoch=10-step=159456.ckpt\"'"
    "uv run python src/predict_gcn.py experiment=swh_1h_multi-head-gat-lstm-vn.yaml data.seq_len=6 data.max_nodes=250 paths.output_dir='/work/ka1176/caroline/gitlab/wave-wind-model-emulator/logs/predict_disjoint_final/swh-gat-lstm-vn-6-250' logger.wandb.id=21vwuxaz 'ckpt_path=\"logs/train/swh_1h_multi-head-gat-lstm-vn/runs/2026-08-21_21-55-20/wwm-emulator-hourly-final/21vwuxaz/checkpoints/epoch=20-step=304416.ckpt\"'"
    "uv run python src/predict_gcn.py experiment=swh_1h_multi-head-gat-lstm-vn.yaml data.seq_len=6 data.max_nodes=50 paths.output_dir='/work/ka1176/caroline/gitlab/wave-wind-model-emulator/logs/predict_disjoint_final/swh-gat-lstm-vn-6-50' logger.wandb.id=nkbd16s5 'ckpt_path=\"logs/train/swh_1h_multi-head-gat-lstm-vn/runs/2026-08-21_21-57-41/wwm-emulator-hourly-final/nkbd16s5/checkpoints/epoch=24-step=362400.ckpt\"'"
    "uv run python src/predict_gcn.py experiment=swh_1h_multi-head-gat-lstm-vn.yaml data.seq_len=48 data.max_nodes=500 paths.output_dir='/work/ka1176/caroline/gitlab/wave-wind-model-emulator/logs/predict_disjoint_final/swh-gat-lstm-vn-48-500' logger.wandb.id=s4x4mgnr 'ckpt_path=\"logs/train/swh_1h_multi-head-gat-lstm-vn/runs/2026-08-22_13-36-30/wwm-emulator-hourly-final/s4x4mgnr/checkpoints/epoch=18-step=271472.ckpt\"'"
    "uv run python src/predict_gcn.py experiment=swh_1h_multi-head-gat-lstm-vn.yaml data.seq_len=48 data.max_nodes=250 paths.output_dir='/work/ka1176/caroline/gitlab/wave-wind-model-emulator/logs/predict_disjoint_final/swh-gat-lstm-vn-48-250' logger.wandb.id=laajpx4l 'ckpt_path=\"logs/train/swh_1h_multi-head-gat-lstm-vn/runs/2026-08-22_01-36-10/wwm-emulator-hourly-final/laajpx4l/checkpoints/epoch=13-step=200032.ckpt\"'"
    "uv run python src/predict_gcn.py experiment=swh_1h_multi-head-gat-lstm-vn.yaml data.seq_len=48 data.max_nodes=50 paths.output_dir='/work/ka1176/caroline/gitlab/wave-wind-model-emulator/logs/predict_disjoint_final/swh-gat-lstm-vn-48-50' logger.wandb.id=pz7ac85k 'ckpt_path=\"logs/train/swh_1h_multi-head-gat-lstm-vn/runs/2026-08-22_04-54-40/wwm-emulator-hourly-final/pz7ac85k/checkpoints/epoch=24-step=357200.ckpt\"'"
    # END
)

eval "${COMMANDS[$SLURM_ARRAY_TASK_ID]}"

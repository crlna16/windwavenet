#!/bin/bash
#SBATCH -A ka1176
#SBATCH -p vader
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH -o scripts/slurm/slurm-%A_%a.out
#SBATCH --array=0-34

COMMANDS=(
    # DenseNet (0)
    #"uv run python src/predict.py experiment=swh_mwp_mwd_1h_mlp test_metrics=true paths.output_dir='/work/ka1176/caroline/gitlab/wave-wind-model-emulator/logs/predict_disjoint_final/swh_mwp_mwd-mlp'
    # LSTM (1-5)
    #"uv run python src/predict.py --config-name=predict_lstm experiment=swh_mwp_mwd_1h_lstm data.seq_len=3 paths.output_dir='/work/ka1176/caroline/gitlab/wave-wind-model-emulator/logs/predict_disjoint_final/swh_mwp_mwd-lstm-3'
    "uv run python src/predict.py --config-name=predict_lstm experiment=swh_mwp_mwd_1h_lstm data.seq_len=6 paths.output_dir='/work/ka1176/caroline/gitlab/wave-wind-model-emulator/logs/predict_disjoint_final/swh_mwp_mwd-lstm-6' logger.wandb.id=9bvw7hi8 'ckpt_path=\"logs/train/swh_mwp_mwd_1h_lstm/runs/2026-08-25_16-11-10/wwm-emulator-hourly-final/9bvw7hi8/checkpoints/epoch=68-step=500112.ckpt\"'"
    #"uv run python src/predict.py --config-name=predict_lstm experiment=swh_mwp_mwd_1h_lstm data.seq_len=12 paths.output_dir='/work/ka1176/caroline/gitlab/wave-wind-model-emulator/logs/predict_disjoint_final/swh_mwp_mwd-lstm-12'
    #"uv run python src/predict.py --config-name=predict_lstm experiment=swh_mwp_mwd_1h_lstm data.seq_len=24 paths.output_dir='/work/ka1176/caroline/gitlab/wave-wind-model-emulator/logs/predict_disjoint_final/swh_mwp_mwd-lstm-24'
    "uv run python src/predict.py --config-name=predict_lstm experiment=swh_mwp_mwd_1h_lstm data.seq_len=48 paths.output_dir='/work/ka1176/caroline/gitlab/wave-wind-model-emulator/logs/predict_disjoint_final/swh_mwp_mwd-lstm-48' TODO"
    # Multi-Head GAT (6-8)
    #"uv run python src/predict_gcn.py experiment=swh_mwp_mwd_1h_multi-head-gat.yaml data.max_nodes=500 paths.output_dir='/work/ka1176/caroline/gitlab/wave-wind-model-emulator/logs/predict_disjoint_final/swh_mwp_mwd-gat-500'
    #"uv run python src/predict_gcn.py experiment=swh_mwp_mwd_1h_multi-head-gat.yaml data.max_nodes=250 paths.output_dir='/work/ka1176/caroline/gitlab/wave-wind-model-emulator/logs/predict_disjoint_final/swh_mwp_mwd-gat-250'
    #"uv run python src/predict_gcn.py experiment=swh_mwp_mwd_1h_multi-head-gat.yaml data.max_nodes=50 paths.output_dir='/work/ka1176/caroline/gitlab/wave-wind-model-emulator/logs/predict_disjoint_final/swh_mwp_mwd-gat-50'
    # Multi-Head GAT VN (9-11)
    "uv run python src/predict_gcn.py experiment=swh_mwp_mwd_1h_multi-head-gat-vn.yaml data.max_nodes=500 paths.output_dir='/work/ka1176/caroline/gitlab/wave-wind-model-emulator/logs/predict_disjoint_final/swh_mwp_mwd-gat-vn-500'  TODO "
    #"uv run python src/predict_gcn.py experiment=swh_mwp_mwd_1h_multi-head-gat-vn.yaml data.max_nodes=250 paths.output_dir='/work/ka1176/caroline/gitlab/wave-wind-model-emulator/logs/predict_disjoint_final/swh_mwp_mwd-gat-vn-250'
    #"uv run python src/predict_gcn.py experiment=swh_mwp_mwd_1h_multi-head-gat-vn.yaml data.max_nodes=50 paths.output_dir='/work/ka1176/caroline/gitlab/wave-wind-model-emulator/logs/predict_disjoint_final/swh_mwp_mwd-gat-vn-50'
    # Multi-Head GAT LSTM VN (12-17)
    "uv run python src/predict_gcn.py experiment=swh_mwp_mwd_1h_multi-head-gat-lstm-vn.yaml data.seq_len=6 data.max_nodes=500 paths.output_dir='/work/ka1176/caroline/gitlab/wave-wind-model-emulator/logs/predict_disjoint_final/swh_mwp_mwd-gat-lstm-vn-6-500'  TODO "
    #"uv run python src/predict_gcn.py experiment=swh_mwp_mwd_1h_multi-head-gat-lstm-vn.yaml data.seq_len=6 data.max_nodes=250 paths.output_dir='/work/ka1176/caroline/gitlab/wave-wind-model-emulator/logs/predict_disjoint_final/swh_mwp_mwd-gat-lstm-vn-6-250'
    "uv run python src/predict_gcn.py experiment=swh_mwp_mwd_1h_multi-head-gat-lstm-vn.yaml data.seq_len=6 data.max_nodes=50 paths.output_dir='/work/ka1176/caroline/gitlab/wave-wind-model-emulator/logs/predict_disjoint_final/swh_mwp_mwd-gat-lstm-vn-6-50' TODO "
    #"uv run python src/predict_gcn.py experiment=swh_mwp_mwd_1h_multi-head-gat-lstm-vn.yaml data.seq_len=48 data.max_nodes=500 paths.output_dir='/work/ka1176/caroline/gitlab/wave-wind-model-emulator/logs/predict_disjoint_final/swh_mwp_mwd-gat-lstm-vn-48-500'
    #"uv run python src/predict_gcn.py experiment=swh_mwp_mwd_1h_multi-head-gat-lstm-vn.yaml data.seq_len=48 data.max_nodes=250 paths.output_dir='/work/ka1176/caroline/gitlab/wave-wind-model-emulator/logs/predict_disjoint_final/swh_mwp_mwd-gat-lstm-vn-48-250'
    #"uv run python src/predict_gcn.py experiment=swh_mwp_mwd_1h_multi-head-gat-lstm-vn.yaml data.seq_len=48 data.max_nodes=50 paths.output_dir='/work/ka1176/caroline/gitlab/wave-wind-model-emulator/logs/predict_disjoint_final/swh_mwp_mwd-gat-lstm-vn-48-50'
    # END
)

eval "${COMMANDS[$SLURM_ARRAY_TASK_ID]}"

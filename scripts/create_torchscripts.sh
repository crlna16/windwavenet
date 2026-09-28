#!/bin/sh
#SBATCH -A ka1176
#SBATCH -p compute
#SBATCH --exclusive
#SBATCH --time=08:00:00

uv run src/export_torchscript.py --output-dir logs/trained_windwavenet_checkpoints/torchscript/

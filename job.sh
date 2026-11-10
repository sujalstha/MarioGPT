#!/bin/bash
#SBATCH --job-name=clt-mario
#SBATCH --account=general
#SBATCH --partition=oucspdn_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=clt-%j.log
set -euo pipefail
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
cd "$HOME/clt-mario"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
env/bin/python -u run.py --phase "${1:-1}"

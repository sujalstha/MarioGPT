#!/bin/bash
set -euo pipefail
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
cd "$HOME/clt-mario"
env/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('distilbert/distilgpt2',local_dir='distilgpt2',allow_patterns=['config.json','generation_config.json','model.safetensors'])"
echo READY

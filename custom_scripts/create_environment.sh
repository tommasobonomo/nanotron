#!/bin/bash

module purge
module load python/3.11.7
module load cuda/12.2


python -m venv /leonardo/home/userexternal/tbonomo0/nanotron/.venv
source /leonardo/home/userexternal/tbonomo0/nanotron/.venv/bin/activate
pip install --upgrade pip wheel setuptools psutil

pip install torch==2.4.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
pip install -e .

pip install transformers wandb
pip install ninja triton "flash-attn>=2.5.0,<2.7.0" --no-build-isolation --no-cache-dir

# TORCH_CUDA_ARCH_LIST="8.0" pip install --no-build-isolation --no-cache-dir /leonardo/home/userexternal/tbonomo0/nanotron/grouped_gemm_binary/nv_grouped_gemm-1.1.4.post3+cu12torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl 
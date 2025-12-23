#!/bin/bash

# tmp dir and W&B
export WANDB_SCRATCH=/leonardo_scratch/large/userexternal/tbonomo0/tmp
mkdir -p "$WANDB_SCRATCH"

# Set all temp dirs env variables to $WANDB_SCRATCH
export TMPDIR=$WANDB_SCRATCH
export TEMP=$WANDB_SCRATCH
export TMP=$WANDB_SCRATCH

export WANDB_MODE=offline
export WANDB_DIR=$WANDB_SCRATCH
export WANDB_CACHE_DIR=$WANDB_SCRATCH/wandb_cache
export WANDB_DATA_DIR=$WANDB_SCRATCH/wandb_data
export WANDB_CONFIG_DIR=$WANDB_SCRATCH/wandb_config

torchrun \
    --nproc_per_node 4 \
    --nnodes $COUNT_NODE \
    --rdzv_backend c10d \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --max_restarts 0 \
    /leonardo/home/userexternal/tbonomo0/nanotron/run_train.py \
    --config-file /leonardo/home/userexternal/tbonomo0/nanotron/examples/config_longctx_continual_128k.yaml
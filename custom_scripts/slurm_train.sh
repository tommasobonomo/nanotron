#!/bin/bash
#SBATCH --job-name=minerva-128k-training
#SBATCH --output=/leonardo/home/userexternal/tbonomo0/nanotron/logs/training/%x-%j.out
#SBATCH --error=/leonardo/home/userexternal/tbonomo0/nanotron/logs/training/%x-%j.err

#SBATCH --account=FAIR_NLP
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=16
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --exclude=lrdn[2000-3450]


module purge
module load nccl/2.22.3-1--gcc--12.2.0-cuda-12.2-spack0.22
module load python/3.11.7

source /leonardo/home/userexternal/tbonomo0/nanotron/.venv/bin/activate

# Multi-node settings
export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=6000
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`
export CUDA_DEVICE_MAX_CONNECTIONS=1

srun /leonardo/home/userexternal/tbonomo0/nanotron/custom_scripts/node_train.sh

# srun bash -c "torchrun \
# --nproc_per_node 4 \
# --nnodes $COUNT_NODE \
# --rdzv_backend c10d \
# --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
# --max_restarts 0 \
# run_train.py --config-file /leonardo/home/userexternal/tbonomo0/nanotron/examples/config_resume_training.yaml"
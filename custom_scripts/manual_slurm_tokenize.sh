#!/bin/bash
#SBATCH --job-name=minerva-tokenize       # Job name
#SBATCH --output=/leonardo/home/userexternal/tbonomo0/nanotron/logs/misc/%x-%j.out         # Name of stdout output file
#SBATCH --error=/leonardo/home/userexternal/tbonomo0/nanotron/logs/misc/%x-%j.err          # Name of stderr error file

#SBATCH -A FAIR_NLP
#SBATCH -p lrd_all_serial
#SBATCH -N 1
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G


module purge
module load nccl/2.22.3-1--gcc--12.2.0-cuda-12.2-spack0.22
module load python/3.11.7

source /leonardo/home/userexternal/tbonomo0/nanotron/.venv/bin/activate


# python tools/preprocess_data.py \
#     --tokenizer-name-or-path /leonardo_scratch/large/userexternal/tbonomo0/models/sapienzanlp--Minerva-7B-base-recipe2/tokenizer.json \
#     --output-folder /leonardo_scratch/large/userexternal/tbonomo0/itagutenberg \
#     --n-tasks 1 \
#     --logging-dir /leonardo/home/userexternal/tbonomo0/nanotron/logs/misc/tokenization \
#     hf \ 
#     --dataset tommasobonomo/ITAGutenberg


python tools/preprocess_data.py \
    --tokenizer-name-or-path /leonardo_scratch/large/userexternal/tbonomo0/models/sapienzanlp--Minerva-7B-base-recipe2/tokenizer.json \
    --output-folder /leonardo_scratch/large/userexternal/tbonomo0/testimole-books \
    --n-tasks 1 \
    --logging-dir /leonardo/home/userexternal/tbonomo0/nanotron/logs/misc/tokenization/testimole-books \
    hf \
    --dataset mrinaldi/abcdefg \
    --subset books

#!/bin/bash
#SBATCH --job-name=minerva-tokenize       # Job name
#SBATCH --output=/leonardo/home/userexternal/tbonomo0/nanotron/logs/misc/%x-%j.out         # Name of stdout output file
#SBATCH --error=/leonardo/home/userexternal/tbonomo0/nanotron/logs/misc/%x-%j.err          # Name of stderr error file

#SBATCH -A FAIR_NLP
#SBATCH -p lrd_all_serial
#SBATCH -N 1
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G


source .venv/bin/activate

python tools/preprocess_data.py \
    --tokenizer-name-or-path /leonardo_scratch/large/userexternal/tbonomo0/models/sapienzanlp--Minerva-7B-base-recipe2/tokenizer.json \
    --output-folder /leonardo_scratch/large/userexternal/tbonomo0/itagutenberg \
    --n-tasks 1 \
    --logging-dir /leonardo/home/userexternal/tbonomo0/nanotron/logs/misc/tokenization \
    hf \
    --dataset tommasobonomo/ITAGutenberg

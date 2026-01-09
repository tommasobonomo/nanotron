#!/bin/bash
#SBATCH --job-name=minerva-download-datasets       # Job name
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

source /leonardo/home/userexternal/tbonomo0/nanotron/.venv/bin/activate

python custom_scripts/download_and_save_to_jsonl.py \
    --dataset-name mrinaldi/abcdefg \
    --subset books \
    --output-dir /leonardo_work/FAIR_NLP/tbonomo0/minerva/data_continual/jsonl/cpt/it/testimole-books \
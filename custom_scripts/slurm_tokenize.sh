#!/bin/bash
#SBATCH --job-name=minerva-tokenize       # Job name
#SBATCH --output=/leonardo/home/userexternal/tbonomo0/nanotron/logs/misc/%x-%j.out         # Name of stdout output file
#SBATCH --error=/leonardo/home/userexternal/tbonomo0/nanotron/logs/misc/%x-%j.err          # Name of stderr error file

#SBATCH -A FAIR_NLP
#SBATCH -p boost_usr_prod
#SBATCH -N 1
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8


module purge
module load nccl/2.22.3-1--gcc--12.2.0-cuda-12.2-spack0.22
module load python/3.11.7

source /leonardo/home/userexternal/tbonomo0/nanotron/.venv/bin/activate


## manu/project_gutenberg en tokenization 
echo "\033[0;32m Tokenizing manu/project_gutenberg en dataset \033[0m"
python tools/preprocess_data.py \
    --tokenizer-name-or-path /leonardo_scratch/large/userexternal/tbonomo0/models/sapienzanlp--Minerva-7B-base-recipe2/tokenizer.json \
    --executor-type local \
    --output-folder /leonardo_work/FAIR_NLP/tbonomo0/minerva/data_continual/tkn/cpt/en/project_gutenberg \
    --n-tasks 52 \
    --logging-dir /leonardo/home/userexternal/tbonomo0/nanotron/logs/misc/tokenization/project_gutenberg_en \
    --name project_gutenberg_en \
    jsonl \
    --dataset /leonardo_work/FAIR_NLP/tbonomo0/minerva/data_continual/jsonl/cpt/en/project_gutenberg
echo "\033[1;32m Finished tokenizing manu/project_gutenberg en dataset \033[0m"

### mrinaldi/abcdefg books tokenization
echo "\033[0;32m Tokenizing mrinaldi/abcdefg books it dataset \033[0m"
python tools/preprocess_data.py \
    --tokenizer-name-or-path /leonardo_scratch/large/userexternal/tbonomo0/models/sapienzanlp--Minerva-7B-base-recipe2/tokenizer.json \
    --executor-type local \
    --output-folder /leonardo_work/FAIR_NLP/tbonomo0/minerva/data_continual/tkn/cpt/it/testimole-books \
    --n-tasks 4 \
    --logging-dir /leonardo/home/userexternal/tbonomo0/nanotron/logs/misc/tokenization/testimole_books_it \
    --name testimole_books_it \
    jsonl \
    --dataset /leonardo_work/FAIR_NLP/tbonomo0/minerva/data_continual/jsonl/cpt/it/testimole-books
echo "\033[1;32m Finished tokenizing mrinaldi/abcdefg books it dataset \033[0m"

### dolmino_math
echo "\033[0;32m Tokenizing dolmino_math en dataset \033[0m"
python tools/preprocess_data.py \
    --tokenizer-name-or-path /leonardo_scratch/large/userexternal/tbonomo0/models/sapienzanlp--Minerva-7B-base-recipe2/tokenizer.json \
    --executor-type local \
    --output-folder /leonardo_work/FAIR_NLP/tbonomo0/minerva/data_continual/tkn/cpt/en/dolmino_math \
    --n-tasks 211 \
    --logging-dir /leonardo/home/userexternal/tbonomo0/nanotron/logs/misc/tokenization/dolmino_math_en \
    --name dolmino_math_en \
    jsonl \
    --dataset /leonardo_work/FAIR_NLP/minerva/data_continual/jsonl/cpt/en/dolmino_math
echo "\033[1;32m Finished tokenizing dolmino_math en dataset \033[0m"

### dolmino_stackexchange
echo "\033[0;32m Tokenizing dolmino_stackexchange en dataset \033[0m"
python tools/preprocess_data.py \
    --tokenizer-name-or-path /leonardo_scratch/large/userexternal/tbonomo0/models/sapienzanlp--Minerva-7B-base-recipe2/tokenizer.json \
    --executor-type local \
    --output-folder /leonardo_work/FAIR_NLP/tbonomo0/minerva/data_continual/tkn/cpt/en/dolmino_stackexchange \
    --n-tasks 25 \
    --logging-dir /leonardo/home/userexternal/tbonomo0/nanotron/logs/misc/tokenization/dolmino_stackexchange_en \
    --name dolmino_stackexchange_en \
    jsonl \
    --dataset /leonardo_work/FAIR_NLP/minerva/data_continual/jsonl/cpt/en/dolmino_stackexchange
echo "\033[1;32m Finished tokenizing dolmino_stackexchange en dataset \033[0m"

### FinePDFs
echo "\033[0;32m Tokenizing FinePDFs en dataset \033[0m"
python tools/preprocess_data.py \
    --tokenizer-name-or-path /leonardo_scratch/large/userexternal/tbonomo0/models/sapienzanlp--Minerva-7B-base-recipe2/tokenizer.json \
    --executor-type local \
    --output-folder /leonardo_work/FAIR_NLP/tbonomo0/minerva/data_continual/tkn/cpt/en/finepdfs \
    --n-tasks 100 \
    --logging-dir /leonardo/home/userexternal/tbonomo0/nanotron/logs/misc/tokenization/finepdfs_en \
    --name finepdfs_en \
    jsonl \
    --dataset /leonardo_work/FAIR_NLP/minerva/data_continual/jsonl/cpt/en/finepdfs
echo "\033[1;32m Finished tokenizing FinePDFs en dataset \033[0m"

### FineWeb EN
echo "\033[0;32m Tokenizing FineWeb en dataset \033[0m"
python tools/preprocess_data.py \
    --tokenizer-name-or-path /leonardo_scratch/large/userexternal/tbonomo0/models/sapienzanlp--Minerva-7B-base-recipe2/tokenizer.json \
    --executor-type local \
    --output-folder /leonardo_work/FAIR_NLP/tbonomo0/minerva/data_continual/tkn/cpt/en/fineweb \
    --n-tasks 40 \
    --logging-dir /leonardo/home/userexternal/tbonomo0/nanotron/logs/misc/tokenization/fineweb_en \
    --name fineweb_en \
    jsonl \
    --dataset /leonardo_work/FAIR_NLP/minerva/data_continual/jsonl/cpt/en/fineweb
echo "\033[1;32m Finished tokenizing FineWeb en dataset \033[0m"

### Flan Tulu
echo "\033[0;32m Tokenizing Flan Tulu en dataset \033[0m"
python tools/preprocess_data.py \
    --tokenizer-name-or-path /leonardo_scratch/large/userexternal/tbonomo0/models/sapienzanlp--Minerva-7B-base-recipe2/tokenizer.json \
    --executor-type local \
    --output-folder /leonardo_work/FAIR_NLP/tbonomo0/minerva/data_continual/tkn/cpt/en/flan_filtered \
    --n-tasks 367 \
    --logging-dir /leonardo/home/userexternal/tbonomo0/nanotron/logs/misc/tokenization/flan_filtered_en \
    --name flan_filtered_en \
    jsonl \
    --dataset /leonardo_work/FAIR_NLP/minerva/data_continual/jsonl/cpt/en/flan_filtered
echo "\033[1;32m Finished tokenizing Flan Tulu en dataset \033[0m"

### The stack EN
echo "\033[0;32m Tokenizing The Stack en dataset \033[0m"
python tools/preprocess_data.py \
    --tokenizer-name-or-path /leonardo_scratch/large/userexternal/tbonomo0/models/sapienzanlp--Minerva-7B-base-recipe2/tokenizer.json \
    --executor-type local \
    --output-folder /leonardo_work/FAIR_NLP/tbonomo0/minerva/data_continual/tkn/cpt/en/the_stack \
    --n-tasks 11 \
    --logging-dir /leonardo/home/userexternal/tbonomo0/nanotron/logs/misc/tokenization/the_stack_en \
    --name the_stack_en \
    jsonl \
    --dataset /leonardo_work/FAIR_NLP/minerva/data_continual/jsonl/cpt/en/the_stack
echo "\033[1;32m Finished tokenizing The Stack en dataset \033[0m"

### Wikipedia EN
echo "\033[0;32m Tokenizing Wikipedia en dataset \033[0m"
python tools/preprocess_data.py \
    --tokenizer-name-or-path /leonardo_scratch/large/userexternal/tbonomo0/models/sapienzanlp--Minerva-7B-base-recipe2/tokenizer.json \
    --executor-type local \
    --output-folder /leonardo_work/FAIR_NLP/tbonomo0/minerva/data_continual/tkn/cpt/en/wikipedia \
    --n-tasks 1 \
    --logging-dir /leonardo/home/userexternal/tbonomo0/nanotron/logs/misc/tokenization/wikipedia_en \
    --name wikipedia_en \
    jsonl \
    --dataset /leonardo_work/FAIR_NLP/data_continual/jsonl/cpt/en/wikipedia
echo "\033[1;32m Finished tokenizing Wikipedia en dataset \033[0m"

### FinePDFs IT
echo "\033[0;32m Tokenizing FinePDFs it dataset \033[0m"
python tools/preprocess_data.py \
    --tokenizer-name-or-path /leonardo_scratch/large/userexternal/tbonomo0/models/sapienzanlp--Minerva-7B-base-recipe2/tokenizer.json \
    --executor-type local \
    --output-folder /leonardo_work/FAIR_NLP/tbonomo0/minerva/data_continual/tkn/cpt/it/finepdfs \
    --n-tasks 50 \
    --logging-dir /leonardo/home/userexternal/tbonomo0/nanotron/logs/misc/tokenization/finepdfs_it \
    --name finepdfs_it \
    jsonl \
    --dataset /leonardo_work/FAIR_NLP/minerva/data_continual/jsonl/cpt/it/finepdfs
echo "\033[1;32m Finished tokenizing FinePDFs it dataset \033[0m"

### Gazzetta IT
echo "\033[0;32m Tokenizing Gazzetta it dataset \033[0m"
python tools/preprocess_data.py \
    --tokenizer-name-or-path /leonardo_scratch/large/userexternal/tbonomo0/models/sapienzanlp--Minerva-7B-base-recipe2/tokenizer.json \
    --executor-type local \
    --output-folder /leonardo_work/FAIR_NLP/tbonomo0/minerva/data_continual/tkn/cpt/it/gazzetta \
    --n-tasks 1 \
    --logging-dir /leonardo/home/userexternal/tbonomo0/nanotron/logs/misc/tokenization/gazzetta_it \
    --name gazzetta_it \
    jsonl \
    --dataset /leonardo_work/FAIR_NLP/minerva/data_continual/jsonl/cpt/it/gazzetta
echo "\033[1;32m Finished tokenizing Gazzetta it dataset \033[0m"

### RedPajamaV2 202314 IT
echo "\033[0;32m Tokenizing RedPajamaV2 2023-14 it dataset \033[0m"
python tools/preprocess_data.py \
    --tokenizer-name-or-path /leonardo_scratch/large/userexternal/tbonomo0/models/sapienzanlp--Minerva-7B-base-recipe2/tokenizer.json \
    --executor-type local \
    --output-folder /leonardo_work/FAIR_NLP/tbonomo0/minerva/data_continual/tkn/cpt/it/redpajamav2/2023-14 \
    --n-tasks 10 \
    --logging-dir /leonardo/home/userexternal/tbonomo0/nanotron/logs/misc/tokenization/redpajamav2_202314_it \
    --name redpajamav2_202314_it \
    jsonl \
    --dataset /leonardo_work/FAIR_NLP/minerva/data_continual/jsonl/cpt/it/redpajamav2/2023-14
echo "\033[1;32m Finished tokenizing RedPajamaV2 2023-14 it dataset \033[0m"

### Wikipedia IT
echo "\033[0;32m Tokenizing Wikipedia it dataset \033[0m"
python tools/preprocess_data.py \
    --tokenizer-name-or-path /leonardo_scratch/large/userexternal/tbonomo0/models/sapienzanlp--Minerva-7B-base-recipe2/tokenizer.json \
    --executor-type local \
    --output-folder /leonardo_work/FAIR_NLP/tbonomo0/minerva/data_continual/tkn/cpt/it/wikipedia \
    --n-tasks 1 \
    --logging-dir /leonardo/home/userexternal/tbonomo0/nanotron/logs/misc/tokenization/wikipedia_it \
    --name wikipedia_it \
    jsonl \
    --dataset /leonardo_work/FAIR_NLP/data_continual/jsonl/cpt/it/wikipedia
echo "\033[1;32m Finished tokenizing Wikipedia it dataset \033[0m"
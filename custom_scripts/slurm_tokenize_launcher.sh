#!/bin/bash
# Launcher script to submit parallel tokenization jobs
# Each dataset gets its own SLURM job for parallel processing

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKER_SCRIPT="$SCRIPT_DIR/slurm_tokenize_worker.sh"
TOKENIZER_PATH="/leonardo_scratch/large/userexternal/tbonomo0/models/sapienzanlp--Minerva-7B-base-recipe2/tokenizer.json"

# Ensure log directories exist
mkdir -p /leonardo/home/userexternal/tbonomo0/nanotron/logs/misc/tokenization

# Function to submit a tokenization job
# Args: name, output_folder, n_tasks, logging_dir, dataset_path, [batch_size], [max_tokens_per_file], [workers]
submit_job() {
    local name=$1
    local output_folder=$2
    local n_tasks=$3
    local logging_dir=$4
    local dataset_path=$5
    local batch_size=${6:-1000}
    local max_tokens_per_file=${7:-100000000}
    local workers=${8:-1}
    
    echo "Submitting job: $name (batch_size=$batch_size, max_tokens=$max_tokens_per_file, workers=$workers)"
    sbatch --job-name="$name" "$WORKER_SCRIPT" \
        "$output_folder" \
        "$n_tasks" \
        "$logging_dir" \
        "$name" \
        "$dataset_path" \
        "$TOKENIZER_PATH" \
        "$batch_size" \
        "$max_tokens_per_file" \
        "$workers"
}

echo "=========================================="
echo "Submitting parallel tokenization jobs..."
echo "=========================================="

# Project Gutenberg EN (long docs)
submit_job "project_gutenberg_en" \
    "/leonardo_work/FAIR_NLP/tbonomo0/minerva/data_continual/tkn/cpt/en/project_gutenberg" \
    4 \
    "/leonardo/home/userexternal/tbonomo0/nanotron/logs/misc/tokenization/project_gutenberg_en" \
    "/leonardo_work/FAIR_NLP/tbonomo0/minerva/data_continual/jsonl/cpt/en/project_gutenberg" \
    100 50000000 1

# Project Gutenberg IT (long docs)
submit_job "project_gutenberg_it" \
    "/leonardo_work/FAIR_NLP/tbonomo0/minerva/data_continual/tkn/cpt/it/project_gutenberg" \
    4 \
    "/leonardo/home/userexternal/tbonomo0/nanotron/logs/misc/tokenization/project_gutenberg_it" \
    "/leonardo_work/FAIR_NLP/tbonomo0/minerva/data_continual/jsonl/cpt/it/itagutenberg" \
    100 50000000 1

# Testimole Books IT (long docs)
submit_job "testimole_books_it" \
    "/leonardo_work/FAIR_NLP/tbonomo0/minerva/data_continual/tkn/cpt/it/testimole-books" \
    4 \
    "/leonardo/home/userexternal/tbonomo0/nanotron/logs/misc/tokenization/testimole_books_it" \
    "/leonardo_work/FAIR_NLP/tbonomo0/minerva/data_continual/jsonl/cpt/it/testimole-books" \
    100 50000000 1

# Academic IT (long docs)
submit_job "academic_it" \
    "/leonardo_work/FAIR_NLP/tbonomo0/minerva/data_continual/tkn/cpt/it/academic" \
    10 \
    "/leonardo/home/userexternal/tbonomo0/nanotron/logs/misc/tokenization/academic_it" \
    "/leonardo_work/FAIR_NLP/tbonomo0/minerva/data_continual/jsonl/cpt/it/academic-ita" \
    100 50000000 1

# Dolmino Math EN (long docs)
submit_job "dolmino_math_en" \
    "/leonardo_work/FAIR_NLP/tbonomo0/minerva/data_continual/tkn/cpt/en/dolmino_math" \
    211 \
    "/leonardo/home/userexternal/tbonomo0/nanotron/logs/misc/tokenization/dolmino_math_en" \
    "/leonardo_work/FAIR_NLP/minerva/data_continual/jsonl/cpt/en/dolmino_math" \
    100 50000000 1

# Dolmino Stackexchange EN
submit_job "dolmino_stackexchange_en" \
    "/leonardo_work/FAIR_NLP/tbonomo0/minerva/data_continual/tkn/cpt/en/dolmino_stackexchange" \
    25 \
    "/leonardo/home/userexternal/tbonomo0/nanotron/logs/misc/tokenization/dolmino_stackexchange_en" \
    "/leonardo_work/FAIR_NLP/minerva/data_continual/jsonl/cpt/en/dolmino_stackexchange"

# FinePDFs EN (long docs)
submit_job "finepdfs_en" \
    "/leonardo_work/FAIR_NLP/tbonomo0/minerva/data_continual/tkn/cpt/en/finepdfs" \
    100 \
    "/leonardo/home/userexternal/tbonomo0/nanotron/logs/misc/tokenization/finepdfs_en" \
    "/leonardo_work/FAIR_NLP/minerva/data_continual/jsonl/cpt/en/finepdfs" \
    100 50000000 1

# FineWeb EN
submit_job "fineweb_en" \
    "/leonardo_work/FAIR_NLP/tbonomo0/minerva/data_continual/tkn/cpt/en/fineweb" \
    40 \
    "/leonardo/home/userexternal/tbonomo0/nanotron/logs/misc/tokenization/fineweb_en" \
    "/leonardo_work/FAIR_NLP/minerva/data_continual/jsonl/cpt/en/fineweb"

# Flan Filtered EN
submit_job "flan_filtered_en" \
    "/leonardo_work/FAIR_NLP/tbonomo0/minerva/data_continual/tkn/cpt/en/flan_filtered" \
    367 \
    "/leonardo/home/userexternal/tbonomo0/nanotron/logs/misc/tokenization/flan_filtered_en" \
    "/leonardo_work/FAIR_NLP/minerva/data_continual/jsonl/cpt/en/flan_filtered"

# The Stack EN
submit_job "the_stack_en" \
    "/leonardo_work/FAIR_NLP/tbonomo0/minerva/data_continual/tkn/cpt/en/the_stack" \
    11 \
    "/leonardo/home/userexternal/tbonomo0/nanotron/logs/misc/tokenization/the_stack_en" \
    "/leonardo_work/FAIR_NLP/minerva/data_continual/jsonl/cpt/en/the_stack"

# Wikipedia EN
submit_job "wikipedia_en" \
    "/leonardo_work/FAIR_NLP/tbonomo0/minerva/data_continual/tkn/cpt/en/wikipedia" \
    1 \
    "/leonardo/home/userexternal/tbonomo0/nanotron/logs/misc/tokenization/wikipedia_en" \
    "/leonardo_work/FAIR_NLP/data_continual/jsonl/cpt/en/wikipedia"

# FinePDFs IT (long docs)
submit_job "finepdfs_it" \
    "/leonardo_work/FAIR_NLP/tbonomo0/minerva/data_continual/tkn/cpt/it/finepdfs" \
    50 \
    "/leonardo/home/userexternal/tbonomo0/nanotron/logs/misc/tokenization/finepdfs_it" \
    "/leonardo_work/FAIR_NLP/minerva/data_continual/jsonl/cpt/it/finepdfs" \
    100 50000000 1

# Gazzetta IT
submit_job "gazzetta_it" \
    "/leonardo_work/FAIR_NLP/tbonomo0/minerva/data_continual/tkn/cpt/it/gazzetta" \
    1 \
    "/leonardo/home/userexternal/tbonomo0/nanotron/logs/misc/tokenization/gazzetta_it" \
    "/leonardo_work/FAIR_NLP/minerva/data_continual/jsonl/cpt/it/gazzetta"

# RedPajamaV2 2023-14 IT
submit_job "redpajamav2_202314_it" \
    "/leonardo_work/FAIR_NLP/tbonomo0/minerva/data_continual/tkn/cpt/it/redpajamav2/2023-14" \
    10 \
    "/leonardo/home/userexternal/tbonomo0/nanotron/logs/misc/tokenization/redpajamav2_202314_it" \
    "/leonardo_work/FAIR_NLP/minerva/data_continual/jsonl/cpt/it/redpajamav2/2023-14"

# Wikipedia IT
submit_job "wikipedia_it" \
    "/leonardo_work/FAIR_NLP/tbonomo0/minerva/data_continual/tkn/cpt/it/wikipedia" \
    1 \
    "/leonardo/home/userexternal/tbonomo0/nanotron/logs/misc/tokenization/wikipedia_it" \
    "/leonardo_work/FAIR_NLP/data_continual/jsonl/cpt/it/wikipedia"

# Benchmarks EN
submit_job "benchmarks_en" \
    "/leonardo_work/FAIR_NLP/tbonomo0/minerva/data_continual/tkn/cpt/en/benchmarks" \
    5 \
    "/leonardo/home/userexternal/tbonomo0/nanotron/logs/misc/tokenization/benchmarks_en" \
    "/leonardo_work/FAIR_NLP/minerva/data_continual/jsonl/cpt/en/benchmarks_ext"

# Benchmarks IT
submit_job "benchmarks_it" \
    "/leonardo_work/FAIR_NLP/tbonomo0/minerva/data_continual/tkn/cpt/it/benchmarks" \
    5 \
    "/leonardo/home/userexternal/tbonomo0/nanotron/logs/misc/tokenization/benchmarks_it" \
    "/leonardo_work/FAIR_NLP/minerva/data_continual/jsonl/cpt/it/benchmarks_ext"

echo "=========================================="
echo "All jobs submitted!"
echo "Use 'squeue -u \$USER' to monitor job status"
echo "=========================================="

#!/bin/bash
#SBATCH --job-name=tokenize-%x          # Job name will be set by launcher
#SBATCH --output=/leonardo/home/userexternal/tbonomo0/nanotron/logs/misc/tokenization/%x-%j.out
#SBATCH --error=/leonardo/home/userexternal/tbonomo0/nanotron/logs/misc/tokenization/%x-%j.err

#SBATCH --account=FAIR_NLP
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G


# Worker script for tokenization - receives parameters from launcher
# Usage: sbatch --job-name=<name> slurm_tokenize_worker.sh <output_folder> <n_tasks> <logging_dir> <name> <dataset_path> [tokenizer_path] [batch_size] [max_tokens_per_file] [workers]
#
# Memory-efficient settings for long documents:
#   - batch_size: 100-500 (default: 1000)
#   - max_tokens_per_file: 50000000-100000000 (default: 100000000)
#   - workers: 1 (default: 1)

OUTPUT_FOLDER=$1
N_TASKS=$2
LOGGING_DIR=$3
NAME=$4
DATASET_PATH=$5
TOKENIZER_PATH=${6:-/leonardo_scratch/large/userexternal/tbonomo0/models/sapienzanlp--Minerva-7B-base-recipe2/tokenizer.json}
BATCH_SIZE=${7:-1000}
MAX_TOKENS_PER_FILE=${8:-100000000}
WORKERS=${9:-1}

if [ -z "$OUTPUT_FOLDER" ] || [ -z "$N_TASKS" ] || [ -z "$LOGGING_DIR" ] || [ -z "$NAME" ] || [ -z "$DATASET_PATH" ]; then
    echo "Error: Missing required arguments"
    echo "Usage: sbatch slurm_tokenize_worker.sh <output_folder> <n_tasks> <logging_dir> <name> <dataset_path> [tokenizer_path] [batch_size] [max_tokens_per_file] [workers]"
    exit 1
fi

module purge
module load nccl/2.22.3-1--gcc--12.2.0-cuda-12.2-spack0.22
module load python/3.11.7

source /leonardo/home/userexternal/tbonomo0/nanotron/.venv/bin/activate

echo "Starting tokenization: $NAME"
echo "  Output folder: $OUTPUT_FOLDER"
echo "  Dataset path: $DATASET_PATH"
echo "  N tasks: $N_TASKS"
echo "  Logging dir: $LOGGING_DIR"
echo "  Tokenizer: $TOKENIZER_PATH"
echo "  Batch size: $BATCH_SIZE"
echo "  Max tokens per file: $MAX_TOKENS_PER_FILE"
echo "  Workers: $WORKERS"

python tools/preprocess_data.py \
    --tokenizer-name-or-path "$TOKENIZER_PATH" \
    --executor-type local \
    --output-folder "$OUTPUT_FOLDER" \
    --n-tasks "$N_TASKS" \
    --logging-dir "$LOGGING_DIR" \
    --name "$NAME" \
    --batch-size "$BATCH_SIZE" \
    --max-tokens-per-file "$MAX_TOKENS_PER_FILE" \
    --workers "$WORKERS" \
    jsonl \
    --dataset "$DATASET_PATH"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "Finished tokenizing $NAME dataset successfully"
else
    echo "ERROR: Tokenization of $NAME failed with exit code $EXIT_CODE"
fi

exit $EXIT_CODE

#!/bin/bash
#SBATCH --job-name=minerva-convert-hf-to-nt        # Job name
#SBATCH --output=/leonardo/home/userexternal/tbonomo0/nanotron/logs/misc/%x-%j.out         # Name of stdout output file
#SBATCH --error=/leonardo/home/userexternal/tbonomo0/nanotron/logs/misc/%x-%j.err          # Name of stderr error file

#SBATCH -A FAIR_NLP
#SBATCH -p boost_usr_prod
#SBATCH -q boost_qos_dbg
#SBATCH -N 1
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --exclusive


module purge
module load nccl/2.22.3-1--gcc--12.2.0-cuda-12.2-spack0.22
module load python/3.11.7

source /leonardo/home/userexternal/tbonomo0/nanotron/.venv/bin/activate

export NT_MODEL_PATH=/leonardo_scratch/large/userexternal/tbonomo0/longctx_checkpoints/15
export HF_MODEL_PATH=/leonardo_scratch/large/userexternal/tbonomo0/models/minerva-7B-base-recipe2-longctx
export TOKENIZER_PATH=/leonardo_scratch/large/userexternal/tbonomo0/models/sapienzanlp--Minerva-7B-base-recipe2/
export MODULE_PATH=examples.mistral.convert_nanotron_to_hf

echo "-----------------------------------"
cmd="torchrun --nproc_per_node=1 -m $MODULE_PATH --checkpoint_path $NT_MODEL_PATH --save_path $HF_MODEL_PATH --tokenizer_name $TOKENIZER_PATH"
echo "COMMAND: $cmd"
echo "-----------------------------------"

eval "$cmd"
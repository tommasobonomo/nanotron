#!/bin/bash
#SBATCH --job-name=minerva-convert-hf-to-nt        # Job name
#SBATCH --output=/leonardo/home/userexternal/tbonomo0/nanotron/logs/misc/%x-%j.out         # Name of stdout output file
#SBATCH --error=/leonardo/home/userexternal/tbonomo0/nanotron/logs/misc/%x-%j.err          # Name of stderr error file

#SBATCH -A FAIR_NLP
#SBATCH -p boost_usr_prod
#SBATCH -q qos_llm_prod
#SBATCH -N 1
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --exclusive

module purge
module load python/3.11.7
module load cuda/12.2

source /leonardo/home/userexternal/tbonomo0/nanotron/.venv/bin/activate

export HF_MODEL_PATH=/leonardo_scratch/large/userexternal/tbonomo0/models/sapienzanlp--Minerva-7B-base-recipe2
export NT_OUTPUT_PATH=/leonardo_scratch/fast/FAIR_NLP/minerva/nanotron_models/minerva-7B-base-recipe2-nt
export SCRIPT_PATH=/leonardo/home/userexternal/tbonomo0/nanotron/examples/mistral/convert_hf_to_nanotron.py
export MODULE_PATH=examples.mistral.convert_hf_to_nanotron

echo "-----------------------------------"
cmd="torchrun --nproc_per_node=1 -m $MODULE_PATH --checkpoint_path $HF_MODEL_PATH --save_path $NT_OUTPUT_PATH"
# cmd="python -m examples.mistral.convert_hf_to_nanotron --checkpoint_path $HF_MODEL_PATH --save_path $NT_OUTPUT_PATH"
echo "COMMAND: $cmd"
echo "-----------------------------------"

eval "$cmd"
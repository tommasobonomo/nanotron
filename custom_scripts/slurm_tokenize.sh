source /leonardo/home/userexternal/tbonomo0/nanotron/.venv/bin/activate

python tools/preprocess_data.py \
    --tokenizer-name-or-path /leonardo_scratch/large/userexternal/tbonomo0/models/sapienzanlp--Minerva-7B-base-recipe2/tokenizer.json \
    --executor-type slurm \
    --output-folder /leonardo_scratch/large/userexternal/tbonomo0/testimole-books \
    --n-tasks 4 \
    --logging-dir /leonardo/home/userexternal/tbonomo0/nanotron/logs/misc/tokenization/testimole-books \
    --name testimole-books \
    hf \
    --dataset mrinaldi/abcdefg \
    --subset books

"""
To process HuggingFace Datasets:
    python3 tools/preprocess_data.py --tokenizer-name-or-path meta-llama/Meta-Llama-3-8B --output-folder datasets/emotion --n-tasks 16 hf --dataset dair-ai/emotion
To process Jsonl files:
    python3 tools/preprocess_data.py --tokenizer-name-or-path meta-llama/Meta-Llama-3-8B --output-folder datasets/c4-es --n-tasks 16 jsonl --dataset raw_datasets/c4-es-json-files

For long documents (to avoid OOM):
    python3 tools/preprocess_data.py --tokenizer-name-or-path meta-llama/Meta-Llama-3-8B --output-folder datasets/long-docs --n-tasks 16 --workers 1 --batch-size 100 --max-tokens-per-file 100000000 jsonl --dataset raw_datasets/long-docs
"""

import argparse

from datatrove.executor.local import LocalPipelineExecutor
from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.readers import HuggingFaceDatasetReader, JsonlReader
from datatrove.pipeline.tokens import DocumentTokenizer


def get_args():
    parser = argparse.ArgumentParser()

    group = parser.add_argument_group(title="Tokenizer")
    group.add_argument(
        "--tokenizer-name-or-path",
        type=str,
        required=True,
        help="A path to a directory containing vocabulary files required by the tokenizer or the model id of a predefined tokenizer hosted inside a model repo on the Hugging Face Hub.",
    )
    group.add_argument(
        "--eos-token",
        type=str,
        default=None,
        help="EOS token to add after each document. Default: None",
    )
    group = parser.add_argument_group(title="Executor")
    group.add_argument(
        "--executor-type",
        type=str,
        default="local",
        choices=["local", "slurm"],
        help="Executor type to run the preprocessing step. Default: local",
    )
    group.add_argument("--mem-per-cpu", type=int, default="7", help="Max RAM available for each CPU core")
    group.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers for local executor. Use 1 for long documents to reduce memory. Default: 1",
    )

    group = parser.add_argument_group(title="Output data")
    group.add_argument(
        "--output-folder", type=str, required=True, help="Path to the output folder to store the tokenized documents"
    )

    group = parser.add_argument_group(title="Tokenization configs")
    group.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for tokenization. Use smaller values (100-500) for long documents to reduce memory. Default: 1000",
    )
    group.add_argument(
        "--max-tokens-per-file",
        type=int,
        default=100_000_000,
        help="Max tokens per output file. Smaller values reduce memory usage. Default: 100000000 (100M)",
    )

    group = parser.add_argument_group(title="Miscellaneous configs")
    group.add_argument(
        "--logging-dir",
        type=str,
        default=None,
        help="Path to a folder for storing the logs of the preprocessing step. Default: None",
    )
    group.add_argument(
        "--n-tasks", type=int, default=8, help="Total number of tasks to run the preprocessing step. Default: 8"
    )
    group.add_argument("--name", type=str, default="dataset", help="Name of dataset that is being tokenized")

    # Subparsers for processing either Hugging Face datasets or jsonl files
    sp = parser.add_subparsers(
        dest="readers",
        required=True,
        description="Type of dataset to process. It can be either a Hugging Face Dataset loaded with datasets.load_data ('hf') or a .jsonl dataset ('jsonl')",
    )

    p1 = sp.add_parser(name="hf")
    p1.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to local stored dataset or repository on the Hugging Face hub that can be loaded with datasets.load_dataset",
    )
    p1.add_argument("--column", type=str, default="text", help="Column to preprocess from the Dataset. Default: text")
    p1.add_argument(
        "--subset",
        type=str,
        default="default",
        help="Name of config (i.e. subset) of the dataset that should be loaded. Default: default",
    )
    p1.add_argument("--split", type=str, default="train", help="Which split of the data to process. Default: train")

    p2 = sp.add_parser(name="jsonl")
    p2.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to a .jsonl file or a folder containing multiple .jsonl files",
    )
    p2.add_argument("--column", type=str, default="text", help="Column to preprocess from the Dataset. Default: text")
    p2.add_argument(
        "--glob-pattern", type=str, default=None, help="A glob pattern to filter files to read. Default: None"
    )

    args = parser.parse_args()

    return args


def main(args):
    # Build datatrove reader
    if args.readers == "hf":
        datatrove_reader = HuggingFaceDatasetReader(
            dataset=args.dataset,
            streaming=True,
            batch_size=100,
            text_key=args.column,
            dataset_options={"split": args.split, "name": args.subset},
        )
    else:
        datatrove_reader = JsonlReader(data_folder=args.dataset, text_key=args.column, glob_pattern=args.glob_pattern)

    pipeline = [
        datatrove_reader,
        DocumentTokenizer(
            output_folder=args.output_folder,
            tokenizer_name_or_path=args.tokenizer_name_or_path,
            eos_token=args.eos_token,
            shuffle_documents=False,
            max_tokens_per_file=args.max_tokens_per_file,
            batch_size=args.batch_size,
        ),
    ]

    if args.executor_type == "slurm":
        executor = SlurmPipelineExecutor(
            pipeline=pipeline,
            tasks=args.n_tasks,
            time="4:00:00",
            sbatch_args={"account": "FAIR_NLP"},
            partition="lrd_all_serial",
            cpus_per_task=1,
            mem_per_cpu_gb=args.mem_per_cpu,
            job_name=f"{args.name}_tokenization",
            venv_path="/leonardo/home/userexternal/tbonomo0/nanotron/.venv",
        )
    elif args.executor_type == "local":
        executor = LocalPipelineExecutor(
            pipeline=pipeline,
            tasks=args.n_tasks,
            logging_dir=args.logging_dir,
            workers=args.workers,
        )
    else:
        raise RuntimeError(f"Unsupported executor {args.executor}")

    executor.run()


if __name__ == "__main__":
    _args = get_args()
    main(_args)

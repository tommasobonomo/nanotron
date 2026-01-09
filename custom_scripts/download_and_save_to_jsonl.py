import argparse
from pathlib import Path

from datasets import IterableDataset, config, load_dataset


def parse_arguments():
    parser = argparse.ArgumentParser(description="Download a dataset and save it to a JSONL file.")
    parser.add_argument(
        "--dataset-name", type=str, required=True, help="The name of the dataset to download from the Hugging Face Hub."
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="The split of the dataset to download (e.g., 'train', 'test', 'validation').",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="default",
        help="The subset of the dataset if applicable.",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="The directory to save the output JSONL files.")
    return parser.parse_args()


def main():
    args = parse_arguments()

    # Create output directory if it doesn't exist
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load the dataset
    config.IN_MEMORY_MAX_SIZE = 1 << 30  # 1GB
    iterable_dataset: IterableDataset = load_dataset(args.dataset_name, args.subset, split=args.split, streaming=True)  # type: ignore

    # Save each shard to a separate JSONL file
    num_shards = iterable_dataset.num_shards
    for i in range(num_shards):
        dataset = iterable_dataset.shard(num_shards=num_shards, index=i)
        output_file = args.output_dir / f"{args.subset}_{args.split}_shard{i}.jsonl"
        dataset.to_json(output_file)

    print(f"Dataset saved to {args.output_dir}")


if __name__ == "__main__":
    main()

import argparse
import logging
import psutil
from pathlib import Path
import os
import json

import huggingface_hub
from datasets import load_dataset
from huggingface_hub import login
from utils import download_streaming_dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

HF_CACHE = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface"))


def main(args):
    token = huggingface_hub.HfFolder.get_token()
    if token is None:
        logger.info("HuggingFace Login")
        login()

    if args.streaming:
        logger.info("==== Starting download CulturaX in streaming mode ====")
        dataset = load_dataset(
            path=args.dataset_name,
            language=args.language,
            split="train",
            streaming=True,
            cache_dir=HF_CACHE,
            token=True,
        )
        download_streaming_dataset(dataset, args)
    else:
        logger.info("==== Starting download CulturaX ====")
        dataset = load_dataset(
            path=args.dataset_name,
            language=args.language,
            # split="train",
            cache_dir=HF_CACHE,
            token=True,
            num_proc=psutil.cpu_count(logical=False),
        )
        # now we can save the dataset in jsonl format, divided in multiple files
        # we would like to parallelize this step, we can use the multiprocessing library
        dataset_path = Path(args.path_to_save)
        dataset_path.mkdir(parents=True, exist_ok=True)

        # create a new file for each split, parallelize this step
        dataset_subsets = []
        for ds_split in dataset:
            print(f"Saving `{ds_split}` split")
            logger.info(f"Saving `{ds_split}` split")
            # if the dataset is too big, we can split it in multiple files
            # but if it is small enough, we can save it in a single file
            shards = dataset[ds_split].num_rows // args.split_size + 1
            print(f"Splitting in {shards} shards")
            for i in range(shards):
                dataset_subsets.append(
                    dataset[ds_split][i * args.split_size : (i + 1) * args.split_size]
                )

            split_path = dataset_path / f"{ds_split}"
            split_path.mkdir(parents=True, exist_ok=True)

            for idx, subset in tqdm(enumerate(dataset_subsets)):
                print(f"Writing to {split_path / f'{idx}.jsonl'}")
                with open(split_path / f"{idx}.jsonl", "w") as f:
                    # it is a dict from key to list of values for that key
                    # e.g. "text" -> [tex1, ..., textn]
                    # "id" -> [id1, ..., idn]
                    # we instead want a list of dicts in the form of
                    # [{"text": text1, "id": id1 }, ..., {"text": textn, "id": idn }]
                    rows = {}
                    for key, values in subset.items():
                        for i, value in enumerate(values):
                            if i not in rows:
                                rows[i] = {}
                            rows[i][key] = value
                    rows = rows.values()
                    f.writelines(json.dumps(r) + "\n" for r in rows)

            dataset_subsets = []


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download HF Dataset")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="uonlp/CulturaX",
        help="Dataset to download.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="it",
        help="Language of the dataset to download.",
    )
    parser.add_argument(
        "--path_to_save",
        type=str,
        default="./",
        help="Path to save the dataset.",
    )
    parser.add_argument(
        "--max_mb",
        type=int,
        default=100,
        help="Max size (in MB) of data to download",
    )
    parser.add_argument(
        "--partial_mb",
        type=int,
        default=10,
        help="Max size (in MB) of data allowed to be in-memory, the size of each saved file",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=10,
        help="Number of document to download to wait every memory check",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Download in streaming mode",
    )
    parser.add_argument(
        "--split_size",
        type=int,
        default=100_000,
        help="Number of document per file during jsonl saving",
    )
    args = parser.parse_args()

    main(args)

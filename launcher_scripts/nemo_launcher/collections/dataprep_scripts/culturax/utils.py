from datasets import Dataset
from tqdm import tqdm
import logging
import os


def save_dataset(dataset, dataset_path, language, suffix):
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    dataset.to_parquet(dataset_path + "/" + language + "_" + str(suffix) + ".parquet")


def download_streaming_dataset(dataset, args):
    dataset_samples = []

    MAX_MEGABYTES = int(args.max_mb)
    PARTIAL_MEGABYTES = int(args.partial_mb)
    hf_dataset = None
    saved_megabytes = 0
    actual_megabytes = 0
    it_ds = 0

    try:
        pbar = tqdm(dataset)
        for i, sample in enumerate(pbar, 1):
            dataset_samples.append(sample)

            if i % int(args.step) == 0:
                hf_dataset = Dataset.from_list(dataset_samples)

                actual_megabytes = hf_dataset.data.nbytes / 1e6  # compute

                pbar.set_description(
                    f"Actual in memory data dimensions: {round(actual_megabytes, 2)} MB | Total data dimensions: {round(saved_megabytes + actual_megabytes, 2)} MB | it: {i}"
                )

                if saved_megabytes + actual_megabytes > MAX_MEGABYTES:
                    print()
                    print("Save final dataset...")
                    save_dataset(hf_dataset, args.dataset_path, args.language, it_ds)
                    break

                if (
                    actual_megabytes > PARTIAL_MEGABYTES
                ):  # the in-memory dataset have to be washed out
                    print()
                    print("Save partial dataset...")

                    save_dataset(hf_dataset, args.dataset_path, args.language, it_ds)

                    # clean stuffs
                    saved_megabytes += actual_megabytes
                    actual_megabytes = 0
                    del dataset_samples
                    del hf_dataset
                    dataset_samples = []

                    it_ds += 1

    except Exception as e:
        print(e)
        print("Trying to save the dataset anyway...")
        save_dataset(hf_dataset, args.dataset_path, args.language, "error")


def get_script_parameters(parser):
    parser.add_argument(
        "-mb",
        "--max-mb",
        dest="max_mb",
        action="store",
        help=f"Max size (in MB) of data to download",
    )
    parser.add_argument(
        "-p",
        "--partial-mb",
        dest="partial_mb",
        action="store",
        help=f"Max size (in MB) of data allowed to be in-memory, the size of each saved file",
    )
    parser.add_argument(
        "-s",
        "--step",
        dest="step",
        action="store",
        default=10,
        help=f"Number of document to download to wait every memory check",
    )
    parser.add_argument(
        "-d",
        "--dataset-path",
        dest="dataset_path",
        action="store",
        help=f"Path location where the dataset will be saved",
    )
    parser.add_argument(
        "-l",
        "--language",
        dest="language",
        action="store",
        help=f"Language contained in CulturaX to download",
    )

    return parser.parse_args()

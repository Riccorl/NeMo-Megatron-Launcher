# # Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.

# """
# Dolly data downloading.
# Example usage:
#  python download.py \
#     --path_to_save=<path/to/save/dolly> \
#     --download_link=<link/to/download>
# """

# import os
# from argparse import ArgumentParser

# from datasets import load_dataset

# dataset_name = "uonlp/CulturaX"


# def get_file_name(link):
#     file_name = link.split("/")[-1]

#     return file_name


# def get_args(dataset_name=dataset_name):
#     parser = ArgumentParser()
#     parser.add_argument(
#         "--path_to_save",
#         type=str,
#         required=True,
#         help="Specify the path where to save the data.",
#     )
#     parser.add_argument(
#         "--dataset_name",
#         type=str,
#         required=False,
#         default=dataset_name,
#         help="Specify the name of the dataset from HF Hub.",
#     )
#     args = parser.parse_args()

#     return args


# def main():
#     args = get_args()
#     path_to_save = args.path_to_save
#     dataset_name = args.dataset_name
#     # link_to_download = args.link_to_download
#     # file_name = get_file_name(link_to_download)

#     print(f"Downloading dataset `{dataset_name}` to {path_to_save} ...")

#     os.system(f"cd {path_to_save}")
#     _ = load_dataset(
#         "uonlp/CulturaX",
#         language="it",
#         token=True,
#         # data_dir="it",
#         # data_files=["it_part_00000.parquet", "it_part_00001.parquet"],
#         split="train[:1%]",
#         cache_dir=path_to_save,
#     )

#     print(f"Dataset `{dataset_name}` was successfully downloaded to {path_to_save} .")


# if __name__ == "__main__":
#     main()

### Import Libraries
import argparse
from datasets import load_dataset
from huggingface_hub import login
import huggingface_hub
# from nemo_launcher.collections.dataprep_scripts.culturax.utils import download_streaming_dataset, get_script_parameters
from utils import download_streaming_dataset, get_script_parameters

### Constant variables
HF_CACHE = "./.hf_cache"


def main(args):
    token = huggingface_hub.HfFolder.get_token()
    if token is None:
        print("HuggingFace Login...")
        login()

    print("==== Starting download CulturaX in streaming mode ====")
    dataset = load_dataset(
        path="uonlp/CulturaX",
        language=args.language,
        split="train",
        streaming=True,
        cache_dir=HF_CACHE,
        token=True,
    )

    download_streaming_dataset(dataset, args)


### Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download CulturaX sample")
    args = get_script_parameters(parser)

    main(args)

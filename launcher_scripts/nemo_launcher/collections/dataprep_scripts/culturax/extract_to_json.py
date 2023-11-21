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

### Import Libraries
import argparse
from datasets import load_dataset
from huggingface_hub import login
import huggingface_hub
from tqdm import tqdm
import json

# from nemo_launcher.collections.dataprep_scripts.culturax.utils import download_streaming_dataset, get_script_parameters
from utils import download_streaming_dataset, get_script_parameters
from pathlib import Path


def main(args):
    # ds = load_dataset(args.input)
    # ds = ds["train"]

    output_path = args.output
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as out:
        for json_file in Path(args.input).glob("*.jsonl"):
            with open(json_file, "r") as f:
                ds = [json.loads(line) for line in f]
            for s in ds:
                out.write(s["text"] + ",")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input")
    parser.add_argument("--output")
    args = parser.parse_args()

    main(args)

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
    ds = load_dataset(args.input)
    ds = ds["train"]

    output_path = args.output
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    splits = ds.num_rows // 10_000
    # create a new file for each 10k rows
    for i in tqdm(range(splits)):
        print(f"Writing to {output_path.with_suffix(f'.{i}.jsonl')}")
        with open(output_path.with_suffix(f".{i}.jsonl"), "w") as f:
            # for subdict in tqdm(ds[i * 10_000 : (i + 1) * 10_000]):
                # it is a dict from key to list of values for that key
                # e.g. "text" -> [tex1, ..., textn]
                # "id" -> [id1, ..., idn]
                # we instead want a list of dicts in the form of
                # [{"text": text1, "id": id1 }, ..., {"text": textn, "id": idn }]
            rows = {}
            for key, values in ds[i * 10_000 : (i + 1) * 10_000].items():
                for i, value in enumerate(values):
                    if i not in rows:
                        rows[i] = {}
                    rows[i][key] = value
            rows = rows.values()
            f.writelines(json.dumps(r) + "\n" for r in rows)

    # with open(args.output, "w") as f:
    #     for row in tqdm(ds):
    #         jsonl_row = huggingface_hub.datasets.Dataset.from_dict(row).to_json()
    #         f.write(jsonl_row + "\n")


### Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download CulturaX sample")
    parser.add_argument("--input")
    parser.add_argument("--output")
    args = parser.parse_args()

    main(args)

#!/bin/bash

if [ -z "DOWNLOAD_DATA_DIR" ]; then
    echo "DOWNLOAD_DATA_DIR not set"
    exit 1
fi

if [ -z "LANGUAGE" ]; then
    echo "LANGUAGE not set"
    exit 1
fi

if [ -z "SPLIT_SIZE" ]; then
    echo "SPLIT_SIZE default to 100_000"
    SPLIT_SIZE=100_000
fi

export PYTHONPATH=/opt/NeMo-Megatron-Launcher/launcher_scripts:${PYTHONPATH}

python nemo_launcher/collections/dataprep_scripts/culturax/download.py \
    --path_to_save $DOWNLOAD_DATA_DIR --language $LANGUAGE --split_size $SPLIT_SIZE

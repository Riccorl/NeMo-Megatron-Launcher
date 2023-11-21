#!/bin/bash

if [ -z "DATA_DIR" ]; then
    echo "DATA_DIR not set"
    exit 1
fi

if [ -z "TOKENIZER_DATA_INPUT" ]; then
    echo "TOKENIZER_DATA_INPUT not set"
    exit 1
fi

export PYTHONPATH=/opt/NeMo-Megatron-Launcher/launcher_scripts:${PYTHONPATH}

python3 /opt/NeMo-Megatron-Launcher/launcher_scripts/main.py --config-name preprocess_data_config.yaml \
    cluster=bcm \
    data_dir=$DATA_DIR \
    tokenizer_data_file=$TOKENIZER_DATA_INPUT \
    cluster_type=bcm

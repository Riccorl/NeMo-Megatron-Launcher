#!/bin/bash

if [ -z "DATA_DIR" ]; then
    echo "DATA_DIR not set"
    exit 1
fi

if [ -z "RAW_DATA_DIR" ]; then
    echo "RAW_DATA_DIR not set"
    exit 1
fi

export PYTHONPATH=/opt/NeMo-Megatron-Launcher/launcher_scripts:${PYTHONPATH}

python3 /opt/NeMo-Megatron-Launcher/launcher_scripts/main.py --config-name tokenizer_training_config.yaml \
    cluster=bcm \
    data_dir=$DATA_DIR \
    raw_data_dir=$RAW_DATA_DIR \
    cluster_type=bcm

#!/usr/bin/env bash

DATASET_DIR=$1
INFERENCE_DATASET_DIR=$2
WORK_DIR=$3
GPUS=$4
PORT=$(shuf -i 1000-9999 -n 1)

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train_active_learning.py $DATASET_DIR \
    $INFERENCE_DATASET_DIR $WORK_DIR --launcher pytorch ${@:5}

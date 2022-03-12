#!/bin/bash
# Train on Arnold US server using all available GPUs

CUDA_LAUNCH_BLOCKING=1 exec python3 $(dirname "$0")/launch.py \
    --launch active_learn.py \
    -a ../configs/_active_learning_/config.yaml \
    $*  # any variable or sequence of variables

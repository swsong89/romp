#!/bin/bash
CUDA_VERSION=cuda-10.2
export PATH=/usr/local/$CUDA_VERSION/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/$CUDA_VERSION/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export CUDA_HOME=/usr/local/$CUDA_VERSION

# source env_romp/bin/activate
conda activate ROMP
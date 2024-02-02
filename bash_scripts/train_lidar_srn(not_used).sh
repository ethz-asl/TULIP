#!/bin/bash

source $HOME/miniconda3/bin/activate
# conda activate plr
# module load eth_proxy python_gpu/3.8.5 gcc/6.3.0
# module load eth_proxy python/3.8.5 gcc/8.2.0 cuda/11.7.0
module load python_gpu/3.10.4 gcc/8.2.0
export PYTHONPATH="${PYTHONPATH}:/cluster/work/riner/users/biyang"
# export TF_ENABLE_ONEDNN_OPTS=0
# export TF_GPU_ALLOCATOR=cuda_malloc_async

python lidar_super_resolution/scripts/run.py
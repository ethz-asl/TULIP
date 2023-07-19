#!/bin/bash

source $HOME/miniconda3/bin/activate
# conda activate plr
module load eth_proxy python_gpu/3.8.5 gcc/6.3.0
export PYTHONPATH="${PYTHONPATH}:/cluster/work/riner/users/biyang"
export TF_ENABLE_ONEDNN_OPTS=0

python lidar_super_resolution/scripts/run.py
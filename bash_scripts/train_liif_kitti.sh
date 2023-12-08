#!/bin/bash

source $HOME/miniconda3/bin/activate
conda activate plr
module load eth_proxy cuda/11.3.1 gcc/8.2.0 ninja
export PYTHONPATH="${PYTHONPATH}:/cluster/work/riner/users/biyang"

args=(
    -c /cluster/work/riner/users/biyang/config/liif_kitti_train.yaml
    -b 32
    )

python /cluster/work/riner/users/biyang/iln/python_src/train_models.py "${args[@]}"
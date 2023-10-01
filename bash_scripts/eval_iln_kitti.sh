#!/bin/bash

source $HOME/miniconda3/bin/activate
conda activate plr
module load eth_proxy cuda/11.3.1 gcc/8.2.0 ninja
export PYTHONPATH="${PYTHONPATH}:/cluster/work/riner/users/biyang"

args=(
    -c /cluster/work/riner/users/biyang/config/iln_kitti_eval.yaml
    -o /cluster/work/riner/users/biyang/experiment/kitti/iln_d1/
    -cp /cluster/work/riner/users/biyang/experiment/kitti/iln_d1/iln_300.pth
    -b 1
    )

python iln/python_src/evaluate_models_on_kitti_dataset.py "${args[@]}"
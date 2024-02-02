#!/bin/bash

source $HOME/miniconda3/bin/activate
conda activate plr
module load eth_proxy cuda/11.3.1 gcc/8.2.0 ninja
export PYTHONPATH="${PYTHONPATH}:/cluster/work/riner/users/biyang"

args=(
    -c /cluster/work/riner/users/biyang/config/liif_durlar_eval.yaml
    -o /cluster/work/riner/users/biyang/experiment/durlar/liif/
    # -cp /cluster/work/riner/users/biyang/experiment/durlar/liif/liif_lidar_340.pth
    -cp /cluster/work/riner/users/biyang/experiment/carla/liif/liif_lidar_400.pth
    -b 1
    )

python iln/python_src/evaluate_models_on_durlar_dataset.py "${args[@]}"
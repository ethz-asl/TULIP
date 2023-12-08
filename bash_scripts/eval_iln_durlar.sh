#!/bin/bash

source $HOME/miniconda3/bin/activate
conda activate plr
module load eth_proxy cuda/11.3.1 gcc/8.2.0 ninja
export PYTHONPATH="${PYTHONPATH}:/cluster/work/riner/users/biyang"

args=(
    -c /cluster/work/riner/users/biyang/config/iln_durlar_eval.yaml
    -o /cluster/work/riner/users/biyang/experiment/durlar/iln_d1/
    #-cp /cluster/work/riner/users/biyang/experiment/durlar/iln_d1/iln_400.pth
    -cp /cluster/work/riner/users/biyang/experiment/carla/iln_d1/iln_400.pth
    -b 1
    )

python iln/python_src/evaluate_models_on_durlar_dataset.py "${args[@]}"
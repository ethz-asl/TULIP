#!/bin/bash

source $HOME/miniconda3/bin/activate
conda activate plr
module load eth_proxy cuda/11.3.1 gcc/8.2.0 ninja
export PYTHONPATH="${PYTHONPATH}:/cluster/work/riner/users/biyang"

args=(
    -c /cluster/work/riner/users/biyang/config/lsr_carla_eval.yaml
    -o /cluster/work/riner/users/biyang/experiment/x16_upsampling/lsr/
    -cp /cluster/work/riner/users/biyang/iln/python_src/models/trained/lsr_8h_2w_400.pth
    -b 1
    )

python iln/python_src/evaluate_models_on_carla_dataset.py "${args[@]}"
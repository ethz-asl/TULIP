#!/bin/bash

source $HOME/miniconda3/bin/activate
conda activate plr
module load eth_proxy cuda/11.3.1 gcc/8.2.0 ninja
export PYTHONPATH="${PYTHONPATH}:/cluster/work/riner/users/biyang"

args=(
    -c /cluster/work/riner/users/biyang/config/lsr_durlar_eval.yaml
    -o /cluster/work/riner/users/biyang/experiment/durlar/lsr/
    -cp /cluster/work/riner/users/biyang/experiment/durlar/lsr/lsr_390.pth
    # -cp /cluster/work/riner/users/biyang/experiment/carla/lsr_20000/lsr_400.pth
    -b 1
    )

python iln/python_src/evaluate_models_on_durlar_dataset.py "${args[@]}"
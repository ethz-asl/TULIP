#!/bin/bash

source $HOME/miniconda3/bin/activate
conda activate plr
module load eth_proxy cuda/11.3.1 gcc/8.2.0 ninja
export PYTHONPATH="${PYTHONPATH}:/cluster/work/riner/users/biyang"

args=(
    -c /cluster/work/riner/users/biyang/iln/python_src/models/lsr_ras20/config/lsr_4h_1w_128_durlar.yaml
    -b 32
    )

python iln/python_src/train_models.py "${args[@]}"
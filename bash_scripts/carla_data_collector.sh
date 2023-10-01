#!/bin/bash

source $HOME/miniconda3/bin/activate
conda activate carla
module load eth_proxy cuda/11.3.1 gcc/8.2.0 ninja
export PYTHONPATH="${PYTHONPATH}:/cluster/work/riner/users/biyang"

args=(
    # --host
    # --port 16
    --world_cofig_file /cluster/home/biyang/ma/carla_simulator/carla_dataset_tools/config/world_config_template.json
    )

python carla_simulator/carla_dataset_tools/data_recorder.py "${args[@]}"
#!/bin/bash

source $HOME/miniconda3/bin/activate
conda activate plr
module load eth_proxy cuda/11.3.1 gcc/8.2.0 ninja



args=(
    --num_data_train 20000
    --num_data_val 2500
    --output_path_name_train train
    --output_path_name_val val
    --input_path ./KITTI/
    --create_val
    )

python /cluster/work/riner/users/biyang/kitti_utils/sample_kitti_dataset.py "${args[@]}"

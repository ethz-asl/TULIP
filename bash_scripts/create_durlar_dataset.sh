#!/bin/bash

source $HOME/miniconda3/bin/activate
conda activate plr
module load eth_proxy cuda/11.3.1 gcc/8.2.0 ninja



args=(
    --data_path /cluster/work/riner/users/biyang/dataset/DurLAR/
    --output_path /cluster/work/riner/users/biyang/dataset/depth_intensity_new/
    --train_data_per_frame 4
    --test_data_per_frame 10
    )

python durlar_utils/create_durlar_dataset.py "${args[@]}"

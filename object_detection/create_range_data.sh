#!/bin/bash

source $HOME/miniconda3/bin/activate
conda activate plr
module load eth_proxy cuda/11.3.1 gcc/8.2.0 ninja
export PYTHONPATH="${PYTHONPATH}:/cluster/work/riner/users/biyang"



args=(
    --data_root /cluster/work/riner/users/biyang/dataset/kitti_object/
    --target_folder_suffix 16x1024
    --downsample
    --split training
    --prefix val
    )

python /cluster/home/biyang/ma/object_detection/PointPillars/get_range_data.py "${args[@]}"

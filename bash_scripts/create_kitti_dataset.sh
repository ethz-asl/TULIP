#!/bin/bash

args=(
    --num_data_train 20000
    --num_data_val 2500
    --output_path_name_train ./rimg_data/train
    --output_path_name_val ./rimg_data/val
    --input_path ./dataset/
    --create_val
    )

python ./kitti_utils/sample_kitti_dataset.py "${args[@]}"

#!/bin/bash

source $HOME/miniconda3/bin/activate
conda activate plr
module load eth_proxy cuda/11.3.1 gcc/8.2.0 ninja
export PYTHONPATH="${PYTHONPATH}:/cluster/work/riner/users/biyang"



args=(
    --data_root /cluster/work/riner/users/biyang/dataset/kitti_object/
    --ckpt /cluster/home/biyang/ma/object_detection/PointPillars/pretrained/epoch_160.pth
    --split val
    --pts_prefix velodyne_16x1024
    --saved_path /cluster/work/riner/users/biyang/dataset/kitti_object/results_/test_low
    )

python /cluster/home/biyang/ma/object_detection/PointPillars/evaluate.py "${args[@]}"

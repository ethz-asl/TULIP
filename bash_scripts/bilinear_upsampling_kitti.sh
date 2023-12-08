#!/bin/bash

source $HOME/miniconda3/bin/activate
conda activate plr
# module load eth_proxy python_gpu/3.8.5 gcc/6.3.0
export PYTHONPATH="${PYTHONPATH}:/cluster/work/riner/users/biyang"
export TF_ENABLE_ONEDNN_OPTS=0


args=(
    --dataset_select kitti
    --data_path_low_res /cluster/work/riner/users/biyang/dataset/KITTI/
    --data_path_high_res /cluster/work/riner/users/biyang/dataset/KITTI/
    --run_name bilinear_upsampling
    --entity biyang
    --project_name kitti_evaluation
    --output_dir /cluster/work/riner/users/biyang/experiment/kitti/bilinear/
    # --wandb_disabled
    --img_size_low_res 16 1024
    --img_size_high_res 64 1024
    --save_pcd
    )

python bilinear/bilinear_upsampling.py "${args[@]}"
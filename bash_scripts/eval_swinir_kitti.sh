#!/bin/bash

source $HOME/miniconda3/bin/activate
conda activate plr
module load eth_proxy cuda/11.3.1 gcc/8.2.0 ninja
export PYTHONPATH="${PYTHONPATH}:/cluster/work/riner/users/biyang"

args=(
    # --task lightweight_sr
    --task classical_sr
    --data_folder /cluster/work/riner/users/biyang/dataset/KITTI/val
    --downsample_factor 4
    --name kitti
    --res '64_1024'
    --phase eval
    --save_dir /cluster/work/riner/users/biyang/experiment/kitti/swinir/
    --save_pcd
    --scale 2
    --in_chans 1
    --model_path /cluster/work/riner/users/biyang/experiment/kitti/swinir/swinir_lidarsr_kitti_x4_16x1024_64x1024/models/960000_G.pth
    --run_name swinir_latest
    --entity biyang
    --project_name kitti_evaluation
    --wandb_disabled
    )
python /cluster/home/biyang/ma/sota_reproduce/SwinIR/main_test_swinir_lidar.py "${args[@]}"
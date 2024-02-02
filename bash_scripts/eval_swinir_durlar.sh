#!/bin/bash

source $HOME/miniconda3/bin/activate
conda activate plr
module load eth_proxy cuda/11.3.1 gcc/8.2.0 ninja
export PYTHONPATH="${PYTHONPATH}:/cluster/work/riner/users/biyang"

args=(
    --task lightweight_sr
    --data_folder /cluster/work/riner/users/biyang/dataset/depth_intensity_new/val/depth/all/
    --downsample_factor 4
    --name durlar
    --res '128_2048'
    --phase eval
    --save_dir /cluster/work/riner/users/biyang/experiment/durlar/swinir/
    # --save_pcd
    --scale 2
    --in_chans 1
    # --model_path /cluster/work/riner/users/biyang/experiment/durlar/swinir/swinir_lidarsr_kitti_x4_32x2048_128x2048/models/395000_E.pth
    --model /cluster/work/riner/users/biyang/experiment/carla/swinir/swinir_lidarsr_kitti_x4_32x2048_128x2048/models/455000_E.pth
    --run_name swinir:train_on_carla_test_on_durlar
    --entity biyang
    --project_name durlar_evaluation
    # --wandb_disabled
    )
python /cluster/home/biyang/ma/sota_reproduce/SwinIR/main_test_swinir_lidar.py "${args[@]}"
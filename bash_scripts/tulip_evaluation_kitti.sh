#!/bin/bash

args=(
    --eval
    --mc_drop
    --noise_threshold 0.03
    --model_select tulip_base
    --pixel_shuffle
    --circular_padding
    --patch_unmerging
    --log_transform
    # Dataset
    --dataset_select kitti
    --data_path_low_res ./dataset/KITTI/
    --data_path_high_res ./dataset/KITTI/
    # --save_pcd
    # WandB Parameters
    --run_name tulip_base
    --entity myentity
    # --wandb_disabled
    --project_name kitti_evaluation
    --output_dir ./trained/tulip_kitti.pth
    --img_size_low_res 16 1024
    --img_size_high_res 64 1024
    --window_size 2 8
    --patch_size 1 4
    --in_chans 1
    )

torchrun --nproc_per_node=1 tulip/main_lidar_upsampling.py "${args[@]}"
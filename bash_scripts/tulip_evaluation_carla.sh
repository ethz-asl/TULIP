#!/bin/bash

args=(

    --eval
    --mc_drop
    --noise_threshold 0.03
    --model_select tulip_base
    --pixel_shuffle
    --circular_padding
    --patch_unmerging
    # Dataset
    --dataset_select carla
    --log_transform
    --data_path_low_res ./dataset/Carla/
    --data_path_high_res ./dataset/Carla/
    # --save_pcd
    # WandB Parameters
    --run_name tulip_base
    --entity myentity
    # --wandb_disabled
    --project_name carla_evaluation
    --output_dir ./trained/tulip_carla.pth
    --img_size_low_res 32 2048
    --img_size_high_res 128 2048
    --window_size 2 8
    --patch_size 1 4
    --in_chans 1
    )

torchrun --nproc_per_node=1 tulip/main_lidar_upsampling.py "${args[@]}"
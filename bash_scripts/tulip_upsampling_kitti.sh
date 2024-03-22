#!/bin/bash


args=(
    --batch_size 8
    --epochs 600
    --num_workers 2
    --lr 5e-4
    --weight_decay 0.01
    --warmup_epochs 60
    # Model parameters
    --model_select tulip_base
    --pixel_shuffle # improve
    --circular_padding # improve
    --log_transform # improve
    --patch_unmerging # improve
    # Dataset
    --dataset_select kitti
    --data_path_low_res ./dataset/KITTI/
    --data_path_high_res ./dataset/KITTI/
    # WandB Parameters
    --run_name tulip_base
    --entity myentity
    # --wandb_disabled
    --project_name experiment_kitti
    --output_dir ./experiment/kitti/tulip_base
    --img_size_low_res 16 1024
    --img_size_high_res 64 1024
    --window_size 2 8
    --patch_size 1 4
    --in_chans 1
    )

# real batch size in training = batch_size * nproc_per_node
torchrun --nproc_per_node=4 tulip/main_lidar_upsampling.py "${args[@]}"
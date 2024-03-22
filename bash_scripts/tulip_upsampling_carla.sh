#!/bin/bash

args=(
    --batch_size 8
    --epochs 600
    --num_workers 2
    --lr 5e-4
    --weight_decay 0.01
    --warmup_epochs 60
    --model_select tulip_large
    --pixel_shuffle # improve
    --circular_padding # improve
    --log_transform # improve
    --patch_unmerging # improve
    # Dataset
    --dataset_select carla
    --data_path_low_res ./dataset/Carla/
    --data_path_high_res /cluster/work/riner/users/biyang/dataset/Carla/
    # WandB Parameters
    --run_name tulip_large
    --entity myentity
    # --wandb_disabled
    --project_name experiment_carla
    --output_dir ./experiment/carla/tulip_large
    # For swim_mae, we have to give the image size that could be split in to 4 windows and then 16x16 patchs
    --img_size_low_res 32 2048
    --img_size_high_res 128 2048
    --window_size 2 8
    --patch_size 1 4
    --in_chans 1
    )

# real batch size in training = batch_size * nproc_per_node
torchrun --nproc_per_node=4 tulip/main_lidar_upsampling.py "${args[@]}"
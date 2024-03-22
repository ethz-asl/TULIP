#!/bin/bash

args=(
    --batch_size 8
    --epochs 600
    --num_workers 2
    --lr 5e-4
    # --save_frequency 10
    --weight_decay 0.01
    --warmup_epochs 60
    --model_select tulip_base
    --pixel_shuffle 
    --circular_padding 
    --log_transform 
    --patch_unmerging 
    # Dataset
    --dataset_select durlar
    --data_path_low_res ./dataset/DurLAR
    --data_path_high_res ./dataset/DurLAR
    # WandB Parameters
    --run_name tulip_base
    --entity myentity
    # --wandb_disabled
    --project_name experiment_durlar
    --output_dir ./experiment/durlar/tulip_base
    --img_size_low_res 32 2048
    --img_size_high_res 128 2048
    --window_size 2 8
    --patch_size 1 4
    --in_chans 1
    )

# real batch size in training = batch_size * nproc_per_node
torchrun --nproc_per_node=4 tulip/main_lidar_upsampling.py "${args[@]}"
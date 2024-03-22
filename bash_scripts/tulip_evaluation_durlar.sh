#!/bin/bash

source $HOME/miniconda3/bin/activate
conda activate plr
module load eth_proxy cuda/11.3.1 gcc/8.2.0 ninja



args=(
    --eval
    --mc_drop
    --noise_threshold 0.0005
    --model_select tulip_base
    --pixel_shuffle
    --circular_padding
    --patch_unmerging
    # Dataset
    --dataset_select durlar
    --log_transform
    --data_path_low_res ./dataset/DurLAR
    --data_path_high_res ./dataset/DurLAR
    # --save_pcd
    # WandB Parameters
    --run_name tulip_base
    --entity myentity
    --wandb_disabled
    --project_name durlar_evaluation
    --output_dir ./experiment/durlar/tulip_base
    --img_size_low_res 32 2048
    --img_size_high_res 128 2048
    --window_size 2 8
    --patch_size 1 4
    --in_chans 1
    )

torchrun --nproc_per_node=1 tulip/main_lidar_upsampling.py "${args[@]}"
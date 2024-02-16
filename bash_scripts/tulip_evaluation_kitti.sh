#!/bin/bash

source $HOME/miniconda3/bin/activate
conda activate plr
module load eth_proxy cuda/11.3.1 gcc/8.2.0 ninja



args=(
    --eval
    --mc_drop
    --noise_threshold 0.03
    --model_select swin_unet
    --pixel_shuffle
    --circular_padding
    --pixel_shuffle_expanding
    --log_transform
    # Dataset
    --dataset_select kitti
    --data_path_low_res /cluster/work/riner/users/biyang/dataset/KITTI/
    --data_path_high_res /cluster/work/riner/users/biyang/dataset/KITTI/
    # --save_pcd
    # WandB Parameters
    --run_name Ours-L:evaluation_within_30_meters
    --entity biyang
    --wandb_disabled
    --project_name kitti_evaluation
    #
    --output_dir /cluster/work/riner/users/biyang/pretrained_weights/tulip_kitti.pth
    --img_size_low_res 16 1024
    --img_size_high_res 64 1024
    --window_size 2 8
    --patch_size 1 4
    --in_chans 1
    )

torchrun --nproc_per_node=1 tulip/main_lidar_upsampling.py "${args[@]}"
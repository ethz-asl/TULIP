#!/bin/bash

source $HOME/miniconda3/bin/activate
conda activate plr
module load eth_proxy cuda/11.3.1 gcc/8.2.0 ninja



args=(
    --batch_size 4
    --epochs 600
    --num_workers 2
    --lr 5e-4
    --weight_decay 0.01
    --warmup_epochs 60
    # Model parameters
    --model_select vit_unet
    --save_frequency 20
    --circular_padding # improve
    --log_transform # improve
    # Dataset
    --dataset_select kitti
    --data_path_low_res /cluster/work/riner/users/biyang/dataset/KITTI/
    --data_path_high_res /cluster/work/riner/users/biyang/dataset/KITTI/
    # WandB Parameters
    --run_name ViTUnet
    --entity biyang
    --wandb_disabled
    --project_name experiment_kitti
    --output_dir /cluster/work/riner/users/biyang/experiment/kitti/Upsampling/AblationStudies/vit_unet_normal
    # For swim_mae, we have to give the image size that could be split in to 4 windows and then 16x16 patchs
    # --img_size_low_res 32 2048
    --img_size_low_res 16 1024
    --img_size_high_res 64 1024
    --patch_size 1 4
    --in_chans 1
    )

torchrun --nproc_per_node=1 tulip/main_lidar_upsampling.py "${args[@]}"
#!/bin/bash

source $HOME/miniconda3/bin/activate
conda activate plr
module load eth_proxy cuda/11.3.1 gcc/8.2.0 ninja



args=(
    --batch_size 8
    --epochs 600
    --num_workers 2
    --lr 5e-4
    # --save_frequency 10
    --weight_decay 0.01
    --warmup_epochs 60
    --model_select tulip_large
    --pixel_shuffle # improve
    --circular_padding # improve
    --log_transform # improve
    --patch_unmerging # improve
    # Dataset
    --dataset_select durlar
    --data_path_low_res /cluster/work/riner/users/biyang/dataset/depth_intensity_new/
    --data_path_high_res /cluster/work/riner/users/biyang/dataset/depth_intensity_new/
    # WandB Parameters
    --run_name tulip_large
    --entity biyang
    # --wandb_disabled
    --project_name experiment_durlar
    --output_dir /cluster/work/riner/users/biyang/experiment/durlar/Upsampling_4/tulip_large
    # For swim_mae, we have to give the image size that could be split in to 4 windows and then 16x16 patchs
    --img_size_low_res 32 2048
    --img_size_high_res 128 2048
    --window_size 2 8
    --patch_size 1 4
    --in_chans 1
    )

# real batch size in training = batch_size * nproc_per_node
torchrun --nproc_per_node=4 tulip/main_lidar_upsampling.py "${args[@]}"
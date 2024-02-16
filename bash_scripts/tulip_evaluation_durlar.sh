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
    # Dataset
    --dataset_select durlar
    --log_transform
    --data_path_low_res /cluster/work/riner/users/biyang/dataset/depth_intensity_new
    --data_path_high_res /cluster/work/riner/users/biyang/dataset/depth_intensity_new
    # --save_pcd
    # WandB Parameters
    --run_name TULIP_normal:evaluation_within_30meters
    --entity biyang
    # --wandb_disabled
    --project_name durlar_evaluation
    --output_dir /cluster/work/riner/users/biyang/experiment/durlar/Upsampling_4/Baseline/
    --img_size_low_res 32 2048
    --img_size_high_res 128 2048
    --window_size 2 8
    --patch_size 1 4
    --in_chans 1
    )

# python mae/main_ouster.py "${args[@]}"   
#python -m torch.distributed.launch --nproc_per_node=2 mae/main_ouster.py "${args[@]}"
torchrun --nproc_per_node=1 tulip/main_lidar_upsampling.py "${args[@]}"
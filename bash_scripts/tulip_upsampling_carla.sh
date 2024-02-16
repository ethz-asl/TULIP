#!/bin/bash

source $HOME/miniconda3/bin/activate
conda activate plr
module load eth_proxy cuda/11.3.1 gcc/8.2.0 ninja



args=(
    --batch_size 8
    --epochs 600
    --num_workers 2
    --lr 5e-4
    --weight_decay 0.01
    --warmup_epochs 60
    --model_select swin_unet
    --pixel_shuffle # improve
    --circular_padding # improve
    --log_transform # improve
    --pixel_shuffle_expanding # improve
    # Dataset
    --dataset_select carla
    --data_path_low_res /cluster/work/riner/users/biyang/dataset/Carla/
    --data_path_high_res /cluster/work/riner/users/biyang/dataset/Carla/
    # WandB Parameters
    --run_name Carla_256_4096:tulip_16x1024
    --entity biyang
    # --wandb_disabled
    --project_name experiment_carla
    --output_dir /cluster/work/riner/users/biyang/experiment/carla/Upsampling/tulip_16x1024_256x4096
    # For swim_mae, we have to give the image size that could be split in to 4 windows and then 16x16 patchs
    --img_size_low_res 32 2048
    --img_size_high_res 128 2048
    --window_size 2 8
    --patch_size 1 4
    --in_chans 1
    )

# python mae/main_ouster.py "${args[@]}"   
#python -m torch.distributed.launch --nproc_per_node=2 mae/main_ouster.py "${args[@]}"
torchrun --nproc_per_node=4 tulip/main_lidar_upsampling.py "${args[@]}"
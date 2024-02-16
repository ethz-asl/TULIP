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
    --dataset_select carla
    --log_transform
    --data_path_low_res /cluster/work/riner/users/biyang/dataset/Carla/
    --data_path_high_res /cluster/work/riner/users/biyang/dataset/Carla/
    --save_pcd
    # WandB Parameters
    --run_name test
    --entity biyang
    # --wandb_disabled
    --project_name carla_evaluation
    --output_dir /cluster/work/riner/users/biyang/experiment/carla/Upsampling/tulip_16x1024_256x4096/
    --img_size_low_res 32 2048
    --img_size_high_res 128 2048
    --window_size 2 8
    --patch_size 1 4
    --in_chans 1
    )

# python mae/main_ouster.py "${args[@]}"   
#python -m torch.distributed.launch --nproc_per_node=2 mae/main_ouster.py "${args[@]}"
torchrun --nproc_per_node=1 tulip/main_lidar_upsampling.py "${args[@]}"
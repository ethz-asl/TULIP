#!/bin/bash

source $HOME/miniconda3/bin/activate
conda activate plr
module load eth_proxy cuda/11.3.1 gcc/8.2.0 ninja



args=(
    --batch_size 32
    --epochs 600
    --num_workers 2
    # --lr 5e-4
    --lr 5e-4
    # --weight_decay 0.0005
    --weight_decay 0.01
    --warmup_epochs 60
    # Model parameters
    --model_select tulip_large
    --pixel_shuffle # improve
    --circular_padding # improve
    --log_transform # improve
    --patch_unmerging # improve
    # Dataset
    --dataset_select kitti
    --data_path_low_res /cluster/work/riner/users/biyang/dataset/KITTI/
    --data_path_high_res /cluster/work/riner/users/biyang/dataset/KITTI/
    # WandB Parameters
    --run_name tulip_large
    --entity biyang
    # --wandb_disabled
    --project_name experiment_kitti
    # Specify the output directory
    --output_dir /cluster/work/riner/users/biyang/experiment/kitti/Upsampling3/tulip_large
    # For swim_mae, we have to give the image size that could be split in to 4 windows and then 16x16 patchs
    --img_size_low_res 16 1024
    --img_size_high_res 64 1024
    --window_size 2 8
    --patch_size 1 4
    --in_chans 1
    )

# python mae/main_ouster.py "${args[@]}"   
# python -m torch.distributed.launch --nproc_per_node=2 mae/main_ouster_upsampling.py "${args[@]}"
torchrun --nproc_per_node=1 tulip/main_lidar_upsampling.py "${args[@]}"
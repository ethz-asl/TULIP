#!/bin/bash

source $HOME/miniconda3/bin/activate
conda activate plr
module load eth_proxy cuda/11.3.1 gcc/8.2.0 ninja



args=(
    --batch_size 32
    --epochs 100
    --num_workers 2
    # --lr 1e-6
    # Model parameters
    --model_select swin_mae
    # Dataset
    --data_path /cluster/work/riner/users/biyang/dataset/depth_intensity_middle
    --crop
    --mask_loss
    # WandB Parameters
    # --run_name swin_mae_crop_5000imgs_075_localpatch2x2_fixedlr1e-6_100epochs
    --run_name localpatch2x2_weighted_loss_liDarReturn_100epochs
    --entity biyang
    --project_name swin_mae_crop_5000imgs_025
    --eval
    --output_dir ./experiment/durlar/swin_mae_centercrop_5000imgs_025_weighted_loss_liDarReturn
    # --wandb_disabled
    --mask_ratio 0.25
    # For swim_mae, we have to give the image size that could be split in to 4 windows and then 16x16 patchs
    --img_size 128 128
    --input_size 128
    --window_size 4
    --patch_size 2 2
    --in_chans 1
    )

# python mae/main_ouster.py "${args[@]}"   
#python -m torch.distributed.launch --nproc_per_node=2 mae/main_ouster.py "${args[@]}"
torchrun --nproc_per_node=4 mae/main_ouster.py "${args[@]}"
#!/bin/bash

source $HOME/miniconda3/bin/activate
conda activate plr
module load eth_proxy cuda/11.3.1 gcc/8.2.0 ninja



args=(
    --batch_size 8
    --epochs 20
    --num_workers 2
    --lr 5e-3
    --weight_decay 0.0005
    # Model parameters
    --eval
    --model_select swin_unet
    # --pretrain /cluster/work/riner/users/biyang/experiment/durlar/LowRes/curriculum_learning_maskratio_075/checkpoint-19.pth
    # Dataset
    --data_path_low_res /cluster/work/riner/users/biyang/dataset/depth_intensity_large_low_res
    --data_path_high_res /cluster/work/riner/users/biyang/dataset/depth_intensity_large
    --save_pcd
    # WandB Parameters
    --run_name upsampling_pretrain_curriculum_learning_squareerror
    --entity biyang
    # --project_name swin_mae_lowres_durlar
    --project_name experiment_upsampling
    # --wandb_disabled
    --output_dir ./experiment/durlar/Upsampling/upsampling_pretrain_curriculum_learning_squareerror
    # For swim_mae, we have to give the image size that could be split in to 4 windows and then 16x16 patchs
    --img_size_low_res 32 2048
    --img_size_high_res 128 2048
    --input_size 128
    --window_size 4
    --patch_size 2 2
    --in_chans 1
    # --img_size 224 224
    # --input_size 224
    # --window_size 7
    # --patch_size 4 4
    # --in_chans 3
    )

# python mae/main_ouster.py "${args[@]}"   
#python -m torch.distributed.launch --nproc_per_node=2 mae/main_ouster.py "${args[@]}"
torchrun --nproc_per_node=4 mae/main_ouster_upsampling.py "${args[@]}"
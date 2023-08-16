#!/bin/bash

source $HOME/miniconda3/bin/activate
conda activate plr
module load eth_proxy cuda/11.3.1 gcc/8.2.0 ninja



args=(
    --batch_size 8
    --epochs 60
    --num_workers 2
    --lr 5e-4
    --weight_decay 0.0005
    --warmup_epochs 10
    # Model parameters
    # --eval
    --model_select swin_unet_v2
    # --edge_loss
    # --pretrain /cluster/work/riner/users/biyang/experiment/durlar/LowRes/linemasking_075_log_transform/checkpoint-59.pth
    --pretrain /cluster/work/riner/users/biyang/pretrained_mae/swinv2_small_patch4_window8_256.pth
    # Dataset
    --data_path_low_res /cluster/work/riner/users/biyang/dataset/depth_intensity_large_low_res
    # --data_path_low_res /cluster/work/riner/users/biyang/dataset/depth_intensity_large
    --data_path_high_res /cluster/work/riner/users/biyang/dataset/depth_intensity_large
    # --log_transform
    # --keep_close_scan
    # --save_pcd
    # WandB Parameters
    --run_name swin_v2_pretrain_imagenet
    --entity biyang
    # --wandb_disabled
    # --project_name swin_mae_lowres_durlar
    --project_name SwinTransformerV2
    # --wandb_disabled
    --output_dir /cluster/work/riner/users/biyang/experiment/durlar/Upsampling_2/SwinV2_Pretrain_ImageNet
    # For swim_mae, we have to give the image size that could be split in to 4 windows and then 16x16 patchs
    # --img_size_low_res 32 2048
    --img_size_low_res 32 2048
    --img_size_high_res 128 2048
    --input_size 128
    # --window_size 4
    # --patch_size 1 4
    --window_size 8
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
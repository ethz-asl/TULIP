#!/bin/bash

source $HOME/miniconda3/bin/activate
conda activate plr
module load eth_proxy cuda/11.3.1 gcc/8.2.0 ninja



args=(
    --batch_size 16
    --epochs 100
    # Model parameters
    --model_select swin_mae
    # --model mae_vit_base_patch16
    # --pretrain "/cluster/work/riner/users/biyang/pretrained_mae/mae_pretrain_vit_base.pth"
    # Dataset
    --data_path /cluster/work/riner/users/biyang/dataset/imagenet-mini/
    # WandB Parameters
    --run_name swin_mae_grayscale_fulldataset025
    --entity biyang
    --project_name TestMAE
    --eval
    --output_dir ./experiment/swim_mae_grayscale_large_025
    # --wandb_disabled
    --mask_ratio 0.25
    --use_cls_token
    # For swim_mae, we have to give the image size that could be split in to 4 windows and then 16x16 patchs
    --img_size 224 224
    --input_size 224
    # Image Net parameters
    --imagenet
    --in_chans 1
    --gray_scale
    )

# python mae/main_ouster.py "${args[@]}"   
#python -m torch.distributed.launch --nproc_per_node=2 mae/main_ouster.py "${args[@]}"
torchrun --nproc_per_node=4 mae/main_ouster.py "${args[@]}"
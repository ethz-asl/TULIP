#!/bin/bash

source $HOME/miniconda3/bin/activate
conda activate plr
module load eth_proxy cuda/11.3.1 gcc/8.2.0 ninja



args=(
    --batch_size 32
    --epochs 1000
    --num_workers 2
    --lr 5e-4
    --weight_decay 0.01
    --warmup_epochs 3 
    --optimizer adamw
    --save_frequency 50
    # --eval
    # Model parameters
    --model_select swin_autoencoder
    --model swin_autoencoder_base
    --circular_padding
    --conv_projection
    --log_transform # I don't think we need this for pretraing, we can simply compute pixel_wise l1 loss, this can be used for upsampling only
    --pixel_shuffle_expanding
    # Dataset
    --dataset_select kitti
    --data_path /cluster/work/riner/users/biyang/dataset/KITTI/
    # --save_pcd
    --loss_on_unmasked
    # WandB Parameters
    --run_name swin_autoencoder
    --entity biyang
    --project_name swin_mae_lowres_kitti
    --wandb_disabled
    --output_dir /cluster/work/riner/users/biyang/experiment/kitti/Lowres/test/
    --mask_ratio 0
    # For swim_mae, we have to give the image size that could be split in to 4 windows and then 16x16 patchs
    --img_size 16 1024
    --input_size 16
    --in_chans 4 # depthwise concatenation
    )

torchrun --nproc_per_node=1 tulip/main_lidar_pretrain.py "${args[@]}"
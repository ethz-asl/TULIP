#!/bin/bash

source $HOME/miniconda3/bin/activate
conda activate plr
module load eth_proxy cuda/11.3.1 gcc/8.2.0 ninja



args=(
    --batch_size 8
    --epochs 100
    --num_workers 2
    --lr 5e-4
    # --seed 300
    # --weight_decay 0.0005
    --weight_decay 0.01
    --warmup_epochs 20
    # Model parameters
    --pretrain /cluster/work/riner/users/biyang/experiment/durlar/LowRes/mr000_128_2048_depthwiseconcat/checkpoint-100.pth
    --analyze
    --model_select swin_unet
    --perceptual_loss
    --pixel_shuffle
    --circular_padding
    --grid_reshape
    --pixel_shuffle_expanding
    # Dataset
    --dataset_select durlar
    --log_transform
    --data_path_low_res /cluster/work/riner/users/biyang/dataset/depth_intensity_new
    --data_path_high_res /cluster/work/riner/users/biyang/dataset/depth_intensity_new
    # WandB Parameters
    --run_name test_perceptual_loss
    --entity biyang
    # --wandb_disabled
    --project_name model_analysis
    --output_dir /cluster/work/riner/users/biyang/experiment/durlar/Upsampling_4/128_2048_mr000pretrain300deepencoder_l2ploss_lamda0.001/
    # For swim_mae, we have to give the image size that could be split in to 4 windows and then 16x16 patchs
    # --img_size_low_res 32 2048
    --img_size_low_res 32 2048
    --img_size_high_res 128 2048
    --input_size 128
    --window_size 4
    --patch_size 1 4
    # --patch_size 4 1
    # --window_size 8
    # --patch_size 4 4
    --in_chans 1
    )

# python mae/main_ouster.py "${args[@]}"   
#python -m torch.distributed.launch --nproc_per_node=2 mae/main_ouster.py "${args[@]}"
torchrun --nproc_per_node=1 /cluster/work/riner/users/biyang/mae/main_ouster_upsampling.py "${args[@]}"
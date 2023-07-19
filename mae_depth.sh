#!/bin/bash

source $HOME/miniconda3/bin/activate
conda activate plr
module load eth_proxy cuda/11.3.1 gcc/8.2.0 ninja



args=(
    --batch_size 32
    --epochs 100
    # Model parameters
    # --eval
    --model_select mae
    --model mae_vit_base_patch8
    --mask_ratio 0.75
    --loss_on_unmasked
    --blr 1e-5 # This could be fixed
    --optimizer adamw
    # --pretrain "/cluster/work/riner/users/biyang/pretrained_mae/mae_pretrain_vit_base.pth"
    # Dataset
    # --data_path /cluster/work/riner/users/biyang/dataset/depth_intensity_middle
    --data_path /cluster/work/riner/users/biyang/dataset/depth_intensity_large
    --crop
    # WandB Parameter
    --run_name mae_075_vit_base_patch8_dataset_large
    --entity biyang
    --project_name MAE_Test2_model_width_and_depth
    --output_dir /cluster/work/riner/users/biyang/experiment/durlar/MAE/datset_size/mae_075_vit_base_patch8_dataset_large
    # --wandb_disabled
    # For swim_mae, we have to give the image size that could be split in to 4 windows and then 16x16 patchs
    --img_size 128 128
    --input_size 128
    --in_chans 1
    )

# python mae/main_ouster.py "${args[@]}"   
#python -m torch.distributed.launch --nproc_per_node=2 mae/main_ouster.py "${args[@]}"
torchrun --nproc_per_node=4 mae/main_ouster.py "${args[@]}"
#!/bin/bash

source $HOME/miniconda3/bin/activate
conda activate plr
module load eth_proxy cuda/11.3.1 gcc/8.2.0 ninja



args=(
    --batch_size 32
    --epochs 20
    --num_workers 2
    # --lr 1e-6
    # Model parameters
    --model_select swin_mae
    --model swin_mae_patch2_base
    --lr 5e-4
    --weight_decay 0.0005
    --optimizer adamw
    --eval
    # --curriculum_learning
    # Dataset
    --data_path /cluster/work/riner/users/biyang/dataset/depth_intensity_large
    --crop
    --loss_on_unmasked
    # --reverse_pixel_value
    # --mask_loss
    # WandB Parameters
    # --run_name swinmae_baseline_maskratio075_20epochs
    --run_name test_visualization
    --entity biyang
    --project_name MAE_Test2_model_width_and_depth
    # --wandb_disabled
    --output_dir /cluster/work/riner/users/biyang/experiment/durlar/SwinMAE/swinmae_curriculum_learning_weighted_reverse_20epochs_lr5e3_customized/
    # --output_dir ./experiment/durlar/test
    --mask_ratio 0.25
    # For swim_mae, we have to give the image size that could be split in to 4 windows and then 16x16 patchs
    --img_size 128 128
    --input_size 128
    --in_chans 1

    # --img_size 224 224
    # --input_size 224
    # --window_size 7
    # --patch_size 4 4
    # --in_chans 3
    )

# python mae/main_ouster.py "${args[@]}"   
#python -m torch.distributed.launch --nproc_per_node=2 mae/main_ouster.py "${args[@]}"
torchrun --nproc_per_node=4 mae/main_ouster.py "${args[@]}"
#!/bin/bash

source $HOME/miniconda3/bin/activate
conda activate plr
module load eth_proxy cuda/11.3.1 gcc/8.2.0 ninja



args=(
    --batch_size 16
    --epochs 20
    --num_workers 2
    --lr 5e-3
    --weight_decay 0.0005
    --optimizer adamw
    # Model parameters
    --curriculum_learning
    --model_select swin_mae
    --model swin_mae_patch2_base
    # --eval
    # Dataset
    --data_path /cluster/work/riner/users/biyang/dataset/depth_intensity_large_low_res
    # --crop
    --loss_on_unmasked
    # WandB Parameters
    --run_name curriculum_learning_maskratio_075
    --entity biyang
    --project_name swin_mae_lowres_durlar
    # --wandb_disabled
    --output_dir ./experiment/durlar/LowRes/curriculum_learning_maskratio_075
    --mask_ratio 0.75
    # For swim_mae, we have to give the image size that could be split in to 4 windows and then 16x16 patchs
    --img_size 32 2048
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
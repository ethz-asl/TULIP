#!/bin/bash

source $HOME/miniconda3/bin/activate
conda activate plr
module load eth_proxy cuda/11.3.1 gcc/8.2.0 ninja



args=(
    --batch_size 1
    --epochs 100
    --num_workers 2
    --lr 5e-4
    # --weight_decay 0.0005
    --weight_decay 0.01
    --warmup_epochs 20
    # Model parameters
    --eval
    --evaluate_with_specific_indices
    #--evaluate_with_different_ranges
    --grid_size 0.1
    --mc_drop
    --noise_threshold 0.03
    # --edge_loss
    --model_select swin_unet_deep
    # --shift_only_leftright
    # --output_multidims
    # --pretrain /cluster/work/riner/users/biyang/experiment/durlar/LowRes/GridReshape_CircularPadding_ConvProjection_ws4/checkpoint-59.pth
    # --pretrain /cluster/work/riner/users/biyang/pretrained_mae/swinv2_small_patch4_window8_256.pth
    # --pretrain_only_encoder
    # --perceptual_loss
    # --pretrain_mae_model swin_mae_patch4_base
    --pixel_shuffle
    --circular_padding
    # --grid_reshape
    --pixel_shuffle_expanding
    --log_transform
    # Dataset
    --dataset_select kitti
    --data_path_low_res /cluster/work/riner/users/biyang/dataset/KITTI/
    --data_path_high_res /cluster/work/riner/users/biyang/dataset/KITTI/
    # --keep_close_scan
    --save_pcd
    # WandB Parameters
    --run_name Kitti64_1024_mcdrop:non_square_window2x8_withall_deepnetwork
    --entity biyang
    --wandb_disabled
    --project_name kitti_evaluation
    --output_dir /cluster/work/riner/users/biyang/experiment/kitti/Upsampling/AblationStudies/non_square_window2x8_withall_deepnetwork/
    # For swim_mae, we have to give the image size that could be split in to 4 windows and then 16x16 patchs
    # --img_size_low_res 32 2048
    --img_size_low_res 16 1024
    --img_size_high_res 64 1024
    --window_size 2 8
    --patch_size 1 4
    # --patch_size 4 1
    # --window_size 8
    # --patch_size 4 4
    --in_chans 1
    )

# python mae/main_ouster.py "${args[@]}"   
#python -m torch.distributed.launch --nproc_per_node=2 mae/main_ouster.py "${args[@]}"
torchrun --nproc_per_node=1 mae/main_ouster_upsampling.py "${args[@]}"
#!/bin/bash

source $HOME/miniconda3/bin/activate
conda activate plr
module load eth_proxy cuda/11.3.1 gcc/8.2.0 ninja



args=(
    --batch_size 8
    --epochs 100
    --num_workers 2
    --lr 5e-4
    # --weight_decay 0.0005
    --weight_decay 0.01
    --warmup_epochs 20
    # Model parameters
    --eval
    # --evaluate_with_specific_indices
    --grid_size 0.1
    --mc_drop
    --noise_threshold 0.0005
    # --output_multidims
    # --edge_loss
    # --model_select swin_unet_moredepths
    --model_select swin_unet
    # --pretrain /cluster/work/riner/users/biyang/experiment/durlar/LowRes/GPC_PSE_LogLoss_Swinv1_100epochs_warmup20_mr075/checkpoint-99.pth
    # --pretrain /cluster/work/riner/users/biyang/pretrained_mae/swinv2_small_patch4_window8_256.pth
    # --pretrain_only_encoder
    # --pretrain_mae_model swin_reversedecoder_base_line_ws4 
    # --perceptual_loss
    --pixel_shuffle
    --circular_padding
    --grid_reshape
    --pixel_shuffle_expanding
    # --delta_pixel_loss
    # --relative_dist_loss
    # Dataset
    --dataset_select durlar
    --log_transform
    --data_path_low_res /cluster/work/riner/users/biyang/dataset/depth_intensity_new
    # --data_path_low_res /cluster/work/riner/users/biyang/dataset/depth_intensity_large
    --data_path_high_res /cluster/work/riner/users/biyang/dataset/depth_intensity_new
    --keep_close_scan
    # --keep_far_scan
    # --save_pcd
    # WandB Parameters
    # --run_name p+c+g+swinv1_deeper+300epcohs+warmup40
    --run_name TULIP_normal:evaluation_within_30meters
    --entity biyang
    # --wandb_disabled
    --project_name durlar_evaluation
    --output_dir /cluster/work/riner/users/biyang/experiment/durlar/Upsampling_4/Baseline/
    # --output_dir /cluster/work/riner/users/biyang/experiment/durlar/Upsampling_4/non_square_window1x16_withall_deepnetwork/
    # --output_dir /cluster/work/riner/users/biyang/experiment/carla/Upsampling/128_2048_baseline/
    # For swim_mae, we have to give the image size that could be split in to 4 windows and then 16x16 patchs
    # --img_size_low_res 32 2048
    --img_size_low_res 32 2048
    --img_size_high_res 128 2048
    --input_size 128
    --window_size 4 4
    --patch_size 1 4
    # --patch_size 4 1
    # --window_size 8
    # --patch_size 4 4
    --in_chans 1
    )

# python mae/main_ouster.py "${args[@]}"   
#python -m torch.distributed.launch --nproc_per_node=2 mae/main_ouster.py "${args[@]}"
torchrun --nproc_per_node=1 mae/main_ouster_upsampling.py "${args[@]}"
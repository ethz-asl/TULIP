#!/bin/bash

source $HOME/miniconda3/bin/activate
conda activate plr
module load eth_proxy cuda/11.3.1 gcc/8.2.0 ninja



args=(
    --batch_size 8
    --epochs 600
    --num_workers 2
    # --lr 5e-4
    --lr 5e-4
    # --weight_decay 0.0005
    # --save_frequency 10
    --weight_decay 0.01
    --warmup_epochs 60
    # --use_ema
    # --output_multidims
    # --feature_weight 0.01
    # Model parameters
    # --eval
    # --mc_drop
    # --edge_loss
    # --model_select swin_unet_moredepths
    --model_select swin_unet_deep
    # --pretrain /cluster/work/riner/users/biyang/experiment/durlar/LowRes/autoencoder_depthwiseconcat_weightedsum/checkpoint-199.pth
    # --pretrain_mae_model swin_autoencoder_base_line_ws4
    # --perceptual_loss
    # --pretrain_only_encoder
    # --fixed_encoder
    # --fixed_all
    --pixel_shuffle # improve
    --circular_padding # improve
    # --grid_reshape # improve
    --log_transform # improve
    # --depth_scale_loss # not improve
    --pixel_shuffle_expanding # improve
    # --relative_dist_loss # not improve
    # --delta_pixel_loss
    # --roll
    # Dataset
    --dataset_select durlar
    --data_path_low_res /cluster/work/riner/users/biyang/dataset/depth_intensity_new/
    --data_path_high_res /cluster/work/riner/users/biyang/dataset/depth_intensity_new/
    # --keep_close_scan
    # --save_pcd
    # WandB Parameters
    --run_name Durlar_128_2048:non_square_window2x8_withall
    --entity biyang
    # --wandb_disabled
    --project_name experiment_durlar
    --output_dir /cluster/work/riner/users/biyang/experiment/durlar/Upsampling_4/non_square_window2x8_withall
    # For swim_mae, we have to give the image size that could be split in to 4 windows and then 16x16 patchs
    # --img_size_low_res 32 2048
    --img_size_low_res 32 2048
    --img_size_high_res 128 2048
    # --input_size 128
    --window_size 2 8
    --patch_size 1 4
    # --patch_size 4 1
    # --window_size 8
    # --patch_size 4 4
    --in_chans 1
    )

# python mae/main_ouster.py "${args[@]}"   
#python -m torch.distributed.launch --nproc_per_node=2 mae/main_ouster.py "${args[@]}"
torchrun --nproc_per_node=4 mae/main_ouster_upsampling.py "${args[@]}"
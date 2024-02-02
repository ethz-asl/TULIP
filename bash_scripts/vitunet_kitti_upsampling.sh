#!/bin/bash

source $HOME/miniconda3/bin/activate
conda activate plr
module load eth_proxy cuda/11.3.1 gcc/8.2.0 ninja



args=(
    --batch_size 4
    --epochs 600
    --num_workers 2
    # --lr 5e-4
    --lr 5e-4
    # --weight_decay 0.0005
    --weight_decay 0.01
    --warmup_epochs 60
    # --feature_weight 0.01
    # Model parameters
    --model_select vit_unet
    --save_frequency 20
    # --resume
    # --output_multidims
    # --pretrain /cluster/work/riner/users/biyang/experiment/kitti/Lowres/autoencoder_depthwiseconcat_weightedsum_noskipcon/checkpoint-499.pth
    # --pretrain_mae_model swin_mae_deepencoder_patch2_base_line_ws4
    # --perceptual_loss
    # --pretrain_only_encoder
    # --shift_only_leftright
    # --pixel_shuffle # improve
    # --circular_padding # improve
    # --grid_reshape # improve
    --log_transform # improve
    # --depth_scale_loss # not improve
    # --pixel_shuffle_expanding # improve
    # --relative_dist_loss # not improve
    # --roll
    # Dataset
    --dataset_select kitti
    --data_path_low_res /cluster/work/riner/users/biyang/dataset/KITTI/
    --data_path_high_res /cluster/work/riner/users/biyang/dataset/KITTI/
    # --keep_close_scan
    # --save_pcd
    # WandB Parameters
    --run_name ViTUnet
    --entity biyang
    # --wandb_disabled
    --project_name experiment_kitti
    --output_dir /cluster/work/riner/users/biyang/experiment/kitti/Upsampling/AblationStudies/vit_unet_normal
    # For swim_mae, we have to give the image size that could be split in to 4 windows and then 16x16 patchs
    # --img_size_low_res 32 2048
    --img_size_low_res 16 1024
    --img_size_high_res 64 1024
    # --input_size 64
    # --window_size 2 8
    --patch_size 1 4
    # --patch_size 4 1
    # --window_size 8
    # --patch_size 4 4
    --in_chans 1
    )

# python mae/main_ouster.py "${args[@]}"   
# python -m torch.distributed.launch --nproc_per_node=2 mae/main_ouster_upsampling.py "${args[@]}"
torchrun --nproc_per_node=4 mae/main_ouster_upsampling.py "${args[@]}"
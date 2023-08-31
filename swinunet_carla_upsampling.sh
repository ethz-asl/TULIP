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
    # --eval
    # --mc_drop
    # --model_select swin_unet_moredepths
    --model_select swin_unet
    # --pretrain /cluster/work/riner/users/biyang/experiment/durlar/LowRes/GridReshape_CircularPadding_ConvProjection_ws4/checkpoint-59.pth
    # --pretrain_only_encoder
    --pixel_shuffle # improve
    --circular_padding # improve
    --grid_reshape # improve
    --log_transform # improve
    # --depth_scale_loss # not improve
    --pixel_shuffle_expanding # improve
    # Dataset
    --dataset_select carla
    --data_path_low_res /cluster/work/riner/users/biyang/dataset/Carla/
    --data_path_high_res /cluster/work/riner/users/biyang/dataset/Carla/
    # --keep_close_scan
    # --save_pcd
    # WandB Parameters
    # --run_name p+c+g+swinv1_deeper+300epcohs+warmup40
    --run_name Carla_256_4096:p+c+g+pse+logloss_swinv1_100epcohs+warmup20
    --entity biyang
    # --wandb_disabled
    --project_name experiment_upsampling_pt4
    --output_dir /cluster/work/riner/users/biyang/experiment/carla/Upsampling/256_4096_GPC_PSE_LogLoss_Swinv1_100epochs_warmup20/
    # For swim_mae, we have to give the image size that could be split in to 4 windows and then 16x16 patchs
    --img_size_low_res 16 1024
    --img_size_high_res 256 4096
    --window_size 4
    --patch_size 1 4
    --in_chans 1
    )

# python mae/main_ouster.py "${args[@]}"   
#python -m torch.distributed.launch --nproc_per_node=2 mae/main_ouster.py "${args[@]}"
torchrun --nproc_per_node=4 mae/main_ouster_upsampling.py "${args[@]}"
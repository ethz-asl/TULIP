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
    # --curriculum_learning
    --model_select swin_autoencoder
    --model swin_mae_patch2_base_line_ws4
    --grid_reshape
    --circular_padding
    --conv_projection
    --log_transform # I don't think we need this for pretraing, we can simply compute pixel_wise l1 loss, this can be used for upsampling only
    --pixel_shuffle_expanding
    # Dataset
    --dataset_select kitti
    --data_path /cluster/work/riner/users/biyang/dataset/KITTI/
    # --save_pcd
    # --crop
    --loss_on_unmasked
    # WandB Parameters
    --run_name swin_autoencoder_kitti200000
    --entity biyang
    --project_name swin_mae_lowres_kitti
    # --wandb_disabled
    --output_dir /cluster/work/riner/users/biyang/experiment/kitti/Lowres/autoencoder_kitti200000/
    --mask_ratio 0
    # For swim_mae, we have to give the image size that could be split in to 4 windows and then 16x16 patchs
    --img_size 16 1024 # 4 x 32
    --input_size 16
    --in_chans 4
    # --img_size 224 224
    # --input_size 224
    # --window_size 7
    # --patch_size 4 4
    # --in_chans 3
    )

# python mae/main_ouster.py "${args[@]}"   
#python -m torch.distributed.launch --nproc_per_node=2 mae/main_ouster.py "${args[@]}"
torchrun --nproc_per_node=4 mae/main_ouster.py "${args[@]}"
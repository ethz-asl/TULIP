#!/bin/bash

source $HOME/miniconda3/bin/activate
conda activate plr
module load eth_proxy cuda/11.3.1 gcc/8.2.0 ninja



args=(
    --batch_size 4
    --epochs 200
    --model mae_vit_base_patch16
    --data_path /cluster/work/riner/users/biyang/dataset/depth_middle/
    --entity biyang
    --project_name Ouster_MAE
    --eval
    --output_dir ./experiment_5000imgs_useclstoken
    # --wandb_disabled
    --mask_ratio 0.75
    --use_cls_token
    )

# python mae/main_ouster.py "${args[@]}"   
#python -m torch.distributed.launch --nproc_per_node=2 mae/main_ouster.py "${args[@]}"
torchrun --nproc_per_node=4 mae/main_ouster.py "${args[@]}"
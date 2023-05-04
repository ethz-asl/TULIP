#!/bin/bash

source $HOME/miniconda3/bin/activate
conda activate plr
module load eth_proxy cuda/11.3.1 gcc/8.2.0 ninja



args=(
    --batch_size 4
    --epochs 1
    --model mae_vit_base_patch16
    --data_path /cluster/work/riner/users/biyang/dataset/depth_test_colored/
    # --input_size 
    # Dataset parameters
    # --local_rank 2
    # --world_size 8
    --pin_mem
    --entity biyang
    --project_name Ouster_MAE
    --eval
    # --output_dir ./experiment_patchsize1632
    # --wandb_disabled
    --mask_ratio 0.25
    )

# python mae/main_ouster.py "${args[@]}"   
#python -m torch.distributed.launch --nproc_per_node=2 mae/main_ouster.py "${args[@]}"
torchrun --nproc_per_node=2 mae/main_ouster.py "${args[@]}"
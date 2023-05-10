#!/bin/bash

source $HOME/miniconda3/bin/activate
conda activate plr
module load eth_proxy cuda/11.3.1 gcc/8.2.0 ninja
# export WANDB_CACHE_DIR='/cluster/work/riner/users/biyang/wandb/.cache'
# export WANDB_DIR='/cluster/work/riner/users/biyang/wandb/'
# export WANDB_CACHE_DIR='/cluster/scratch/biyang/.cache'

args=(
    --config ./config/multimae_ouster.yaml
    )

# python mae/main_ouster.py "${args[@]}"   
#python -m torch.distributed.launch --nproc_per_node=2 mae/main_ouster.py "${args[@]}"
torchrun --nproc_per_node=2 MultiMAE/run_ouster.py "${args[@]}"
#!/bin/bash

source $HOME/miniconda3/bin/activate
conda activate plr
module load eth_proxy cuda/11.3.1 gcc/8.2.0 ninja
export PYTHONPATH="${PYTHONPATH}:/cluster/work/riner/users/biyang"

args=(
    --opt /cluster/home/biyang/ma/config/train_swinir_sr_carla.json
    --dist True
    )



torchrun --nproc_per_node=4 --master_port=1234 /cluster/home/biyang/ma/sota_reproduce/KAIR/main_train_psnr.py "${args[@]}"
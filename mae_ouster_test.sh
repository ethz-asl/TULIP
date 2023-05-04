#!/bin/bash

source $HOME/miniconda3/bin/activate
conda activate plr
module load eth_proxy cuda/11.3.1 gcc/8.2.0 ninja


###!! local rank and nproc_per_node should remain same, also the number of gpus in sbatch submission
args=(
    --batch_size 4
    --epochs 2
    --model mae_vit_base_patch16
    --data_path /cluster/work/riner/users/biyang/dataset/depth_test_colored/
    # --input_size 
    # Dataset parameters
    # --local_rank 4
    # --world_size 2
    --pin_mem
    --entity biyang
    --project_name Ouster_MAE
    --eval
    --load_model "/cluster/work/riner/users/biyang/output_dir/checkpoint-1.pth"
    # --wandb_disabled
    )

python mae/test_ouster.py "${args[@]}"
#python -m torch.distributed.launch --nproc_per_node=1 mae/test_ouster.py "${args[@]}"
# torchrun mae/test_ouster.py "${args[@]}"
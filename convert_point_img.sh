#!/bin/bash

source $HOME/miniconda3/bin/activate
conda activate plr
module load eth_proxy cuda/11.3.1 gcc/8.2.0 ninja



args=(
    --input_path /cluster/work/riner/users/biyang/dataset/DurLAR/DurLAR_20210716/
    # --color
    --normalize
    )

python durlar_utils/get_intensity_depth.py "${args[@]}"

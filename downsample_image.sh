#!/bin/bash

source $HOME/miniconda3/bin/activate
conda activate plr
module load eth_proxy cuda/11.3.1 gcc/8.2.0 ninja



args=(
    --input_path /cluster/work/riner/users/biyang/dataset/depth_intensity_middle
    --output_path_name depth_intensity_middle_low_res
    --downsampling_factor 4
    )

python durlar_utils/preprocess_data.py "${args[@]}"

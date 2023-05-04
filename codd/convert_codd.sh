#!/bin/bash

source $HOME/miniconda3/bin/activate
conda activate plr
module load eth_proxy cuda/11.3.1 gcc/8.2.0 ninja



args=(
    --mode val
    #--save_data
    --output_folder range_intensity
    --input_path /cluster/scratch/biyang/data
    )

python convert_codd_to_range_map.py "${args[@]}"



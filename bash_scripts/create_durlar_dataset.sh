#!/bin/bash

source $HOME/miniconda3/bin/activate
conda activate plr
module load eth_proxy cuda/11.3.1 gcc/8.2.0 ninja



args=(
    --input_path ./DurLAR/
    --output_path_name_train train
    --output_path_name_val val
    --train_data_per_frame 4
    --test_data_per_frame 10
    --create_val
    )

python durlar_utils/sample_durlar_dataset.py "${args[@]}"

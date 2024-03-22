#!/bin/bash

args=(
    --input_path ./DurLAR/
    --output_path_name_train train
    --output_path_name_val val
    --train_data_per_frame 4
    --test_data_per_frame 10
    --create_val
    )

python durlar_utils/sample_durlar_dataset.py "${args[@]}"

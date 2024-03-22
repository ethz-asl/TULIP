import numpy as np
import os
import argparse
import cv2
from glob import glob
import pathlib
import random

import shutil


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_data_train', type=int, default=21000)
    parser.add_argument('--num_data_val', type=int, default=2500)
    parser.add_argument("--input_path", type=str , default="/cluster/work/riner/users/biyang/dataset/KITTI/")
    parser.add_argument("--output_path_name_train", type = str, default = "kitti_train")
    parser.add_argument("--output_path_name_val", type = str, default = "kitti_val")
    parser.add_argument("--create_val", action='store_true', default=False)
   
    return parser.parse_args()


def create_range_map(points_array, image_rows_full, image_cols, ang_start_y, ang_res_y, ang_res_x, max_range, min_range):
    range_image = np.zeros((image_rows_full, image_cols, 1), dtype=np.float32)
    intensity_map = np.zeros((image_rows_full, image_cols, 1), dtype=np.float32)
    x = points_array[:,0]
    y = points_array[:,1]
    z = points_array[:,2]
    intensity = points_array[:, 3]
    # find row id

    vertical_angle = np.arctan2(z, np.sqrt(x * x + y * y)) * 180.0 / np.pi
    relative_vertical_angle = vertical_angle + ang_start_y
    rowId = np.int_(np.round_(relative_vertical_angle / ang_res_y))
    # Inverse sign of y for kitti data
    horitontal_angle = np.arctan2(x, y) * 180.0 / np.pi

    colId = -np.int_((horitontal_angle-90.0)/ang_res_x) + image_cols/2

    shift_ids = np.where(colId>=image_cols)
    colId[shift_ids] = colId[shift_ids] - image_cols
    colId = colId.astype(np.int64)
    # filter range
    thisRange = np.sqrt(x * x + y * y + z * z)
    thisRange[thisRange > max_range] = 0
    thisRange[thisRange < min_range] = 0

    # filter Internsity
    intensity[thisRange > max_range] = 0
    intensity[thisRange < min_range] = 0


    valid_scan = (rowId >= 0) & (rowId < image_rows_full) & (colId >= 0) & (colId < image_cols)

    rowId_valid = rowId[valid_scan]
    colId_valid = colId[valid_scan]
    thisRange_valid = thisRange[valid_scan]
    intensity_valid = intensity[valid_scan]

    range_image[rowId_valid, colId_valid, :] = thisRange_valid.reshape(-1, 1)
    intensity_map[rowId_valid, colId_valid, :] = intensity_valid.reshape(-1, 1)

    lidar_data_projected = np.concatenate((range_image, intensity_map), axis = -1)

    return lidar_data_projected


def load_from_bin(bin_path):
    lidar_data = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    # ignore reflectivity info
    return lidar_data

def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines

def main(args):
    num_data_train = args.num_data_train
    num_data_val = args.num_data_val
    dir_name = os.path.dirname(args.input_path)
    output_dir_name_train = os.path.join(dir_name, args.output_path_name_train)
    pathlib.Path(output_dir_name_train).mkdir(parents=True, exist_ok=True)
    if args.create_val:
        output_dir_name_val = os.path.join(dir_name, args.output_path_name_val)
        pathlib.Path(output_dir_name_val).mkdir(parents=True, exist_ok=True)

    train_split_path = "./kitti_utils/train_files.txt"
    val_split_path = "./kitti_utilsval_files.txt"

    train_split = np.array(readlines(train_split_path), dtype = str)
    val_split = np.array(readlines(val_split_path), dtype = str)

    train_data = []
    val_data = []

    # If the required data number is lower than the total number of scan, then sample the scan
    if num_data_train < len(train_split):
        train_split = np.random.choice(train_split, num_data_train, replace=False)
        for train_folder in train_split:
            sample_one_train_data = np.random.choice(np.array(glob(os.path.join(dir_name, train_folder, "velodyne_points/data/*.bin"))), 1, replace=False)
            train_data.append(sample_one_train_data[0])

    # If the required data number is higher than the total number of scan
    else:
        sample_data_per_scan = num_data_train // len(train_split) + 1
        for train_folder in train_split:
            sample_one_train_data = np.random.choice(np.array(glob(os.path.join(dir_name, train_folder, "velodyne_points/data/*.bin"))), sample_data_per_scan, replace=False)
            train_data += list(sample_one_train_data)

        random.shuffle(train_data)
        train_data = train_data[:num_data_train]
        
        
    assert len(train_data) == num_data_train, "The number of training data is not correct"  


    if args.create_val:
        if num_data_val < len(val_split):
            val_split = np.random.choice(val_split, num_data_val, replace=False)
            for val_folder in val_split:
                sample_one_val_data = np.random.choice(np.array(glob(os.path.join(dir_name, val_folder, "velodyne_points/data/*.bin"))), 1, replace=False)
                val_data.append(sample_one_val_data[0])
        else:
            sample_data_per_scan = num_data_val // len(val_split) + 1
            for val_folder in val_split:
                sample_one_val_data = np.random.choice(np.array(glob(os.path.join(dir_name, val_folder, "velodyne_points/data/*.bin"))), sample_data_per_scan, replace=False)
                val_data += list(sample_one_val_data)
            
            random.shuffle(val_data)
            val_data = val_data[:num_data_val]

        assert len(val_data) == num_data_val, "The number of validation data is not correct"


    image_rows = 64
    image_cols = 1024
    ang_start_y = 24.8
    ang_res_y = 26.8 / (image_rows -1)
    ang_res_x = 360 / image_cols
    max_range = 120
    min_range = 0
    
    
    # Move the data to the output directory
    for i, train_data_path in enumerate(train_data):

        lidar_data = load_from_bin(train_data_path)
        range_intensity_map = create_range_map(lidar_data, image_rows_full = image_rows, image_cols = image_cols, ang_start_y = ang_start_y, ang_res_y = ang_res_y, ang_res_x = ang_res_x, max_range = max_range, min_range = min_range)

        np.save(os.path.join(output_dir_name_train,'{:08d}.npy'.format(i)), range_intensity_map.astype(np.float32))

    if args.create_val:
        for j, val_data_path in enumerate(val_data):
            lidar_data = load_from_bin(val_data_path)
            range_intensity_map = create_range_map(lidar_data, image_rows_full = image_rows, image_cols = image_cols, ang_start_y = ang_start_y, ang_res_y = ang_res_y, ang_res_x = ang_res_x, max_range = max_range, min_range = min_range)
            np.save(os.path.join(output_dir_name_val,'{:08d}.npy'.format(j)), range_intensity_map.astype(np.float32))

    

if __name__ == "__main__":
    args = read_args()
    main(args)
    
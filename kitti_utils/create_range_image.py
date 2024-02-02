import numpy as np
import os
import argparse
# import cv2
from glob import glob
# import pathlib
# import random

# import shutil


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str ,required = True)
    parser.add_argument("--output_path", type = str, required = True)
    parser.add_argument("--high_res", type = bool, default = False)
   
    return parser.parse_args()


def create_range_map(points_array, image_rows_full, image_cols, ang_start_y, ang_res_y, ang_res_x, max_range, min_range):
    #print('processing {}th point cloud message...\r'.format(range_image_array.shape[0])),
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
    # find column id
    # Inverse sign of y for kitti data
    horitontal_angle = np.arctan2(x, y) * 180.0 / np.pi
    # horitontal_angle = np.arctan2(-y, x) * 180.0 / np.pi

    colId = -np.int_((horitontal_angle-90.0)/ang_res_x) + image_cols/2
    # colId = -np.int_(horitontal_angle / ang_res_x) + image_cols / 2

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

    # # save range info to range image
    # for i in range(len(thisRange)):
    #     if rowId[i] < 0 or rowId[i] >= image_rows_full or colId[i] < 0 or colId[i] >= image_cols:
    #         continue
    #     range_image[rowId[i], colId[i], :] = thisRange[i]
    #     intensity_map[rowId[i], colId[i], :] = intensity[i]

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



    input_files = glob(args.input_path + '/*.bin')
    # print(len(input_files))
    input_files.sort()

    # print(len(val_data))
    # print(len(train_data))

    image_rows = 64
    image_cols = 1024
    ang_start_y = 24.8
    # ang_start_y = 25
    ang_res_y = 26.8 / (image_rows -1)
    ang_res_x = 360 / image_cols
    max_range = 120
    min_range = 0
    
    


    # Create the output directory
    
    
    # Move the data to the output directory
    for i, data in enumerate(input_files):
        
        name = os.path.basename(data)

        lidar_data = load_from_bin(data)
        range_intensity_map = create_range_map(lidar_data, image_rows_full = image_rows, image_cols = image_cols, ang_start_y = ang_start_y, ang_res_y = ang_res_y, ang_res_x = ang_res_x, max_range = max_range, min_range = min_range)

        range_intensity_map.astype(np.float32).tofile(os.path.join(args.output_path, name))
        # np.save(os.path.join(args.output_path, name), range_intensity_map.astype(np.float32))
        # shutil.copy(train_data_path, os.path.join(output_dir_name_train,'{:08d}.npy'.format(i)))

    # if args.create_val:
    #     for j, val_data_path in enumerate(val_data):
    #         # shutil.copy(val_data_path, os.path.join(output_dir_name_val,'{:08d}.npy'.format(j)))
    #         lidar_data = load_from_bin(val_data_path)
    #         range_intensity_map = create_range_map(lidar_data, image_rows_full = image_rows, image_cols = image_cols, ang_start_y = ang_start_y, ang_res_y = ang_res_y, ang_res_x = ang_res_x, max_range = max_range, min_range = min_range)
    #         np.save(os.path.join(output_dir_name_val,'{:08d}.bin'.format(j)), range_intensity_map.astype(np.float32))

    








if __name__ == "__main__":
    args = read_args()
    main(args)
    
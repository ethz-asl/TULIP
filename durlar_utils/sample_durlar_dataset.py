import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import os
import argparse
from bin_to_img import *
import pathlib
from glob import glob

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type = int, default = 128)
    parser.add_argument("--cols", type = int, default = 2048)
    parser.add_argument("--max_range", type = int, default = 128)
    parser.add_argument('--range', nargs="+", type=int, help='start and end frame number')

    parser.add_argument("--input_path", type=str , default=None)
    parser.add_argument("--train_data_per_frame", type = int, default = 4, help = "skip rate of training data")
    parser.add_argument("--test_data_per_frame", type = int, default = 10, help = "skip rate of test data")
    parser.add_argument("--output_path_name_train", type = str, default = "durlar_train")
    parser.add_argument("--output_path_name_val", type = str, default = "durlar_val")
    parser.add_argument("--create_val", action='store_true', default=False)

    return parser.parse_args()


def main(args):
    # Default train-test split setting
    train_data_folder = ['DurLAR_20210716', 'DurLAR_20211012', 'DurLAR_20211208', 'DurLAR_20210901']
    test_data_folder = ['DurLAR_20211209']

    train_data_per_frame = args.train_data_per_frame
    test_data_per_frame = args.test_data_per_frame

    # Create output paths
    dir_name = os.path.dirname(args.input_path)
    output_dir_name_train = os.path.join(dir_name, args.output_path_name_train)
    pathlib.Path(output_dir_name_train).mkdir(parents=True, exist_ok=True)
    if args.create_val:
        output_dir_name_val = os.path.join(dir_name, args.output_path_name_val)
        pathlib.Path(output_dir_name_val).mkdir(parents=True, exist_ok=True)


    # Load all test data (fullpath name)
    train_data = []
    for folder in train_data_folder:
        pcd_files = glob(os.path.join(args.input_path, folder, "ouster_points/data/*.bin"))

        pcd_files.sort()
        train_data.extend(pcd_files)


    # Load all test data (fullpath name)
    test_data = []
    for folder in test_data_folder:
        pcd_files = glob(os.path.join(args.input_path, folder, "ouster_points/data/*.bin"))
        pcd_files.sort()
        test_data.extend(pcd_files)


    # Copy the data to the output folder and rename it
    print("There are totally {} data for training, we skip with rate {}".format(len(train_data), train_data_per_frame))
    print("There are totally {} data for testing, we skip with rate {}".format(len(test_data), test_data_per_frame))



    # Saving Training data
    for i in range(len(train_data)):
        if i % train_data_per_frame == 0:

            scan = (np.fromfile(train_data[i], dtype=np.float32)).reshape(-1, 4)
            range_map, intensity = pcd_to_img(scan=scan, rows=args.rows, cols = args.cols)
            range_intensity_map = np.concatenate((range_map[..., None], intensity[..., None]), axis = -1)
            np.save(os.path.join(output_dir_name_train,'{:08d}.npy'.format(i)), range_intensity_map.astype(np.float32))

    print("Training Data saved!")

    if args.create_val:
        for i in range(len(test_data)):
            if i % test_data_per_frame == 0:
                scan = (np.fromfile(test_data[i], dtype=np.float32)).reshape(-1, 4)
                range_map, intensity = pcd_to_img(scan=scan, rows=args.rows, cols = args.cols)
                range_intensity_map = np.concatenate((range_map[..., None], intensity[..., None]), axis = -1)

                np.save(os.path.join(output_dir_name_val,'{:08d}.npy'.format(i)), range_intensity_map.astype(np.float32))


        print("Test Data saved!")
    


if __name__ == "__main__":
    args = read_args()
    main(args)
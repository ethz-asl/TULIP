import torch
import trimesh
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import os
import argparse
import cv2
from glob import glob
import pathlib


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=["val", "train", "test"])
    parser.add_argument("--input_path", type=str , default=None)
    parser.add_argument("--output_path_name", type = str, default = None)


    parser.add_argument("--downsampling_factor", type = int, default = 4)
    # parser.add_argument("--cols", type = int, default = 2048)
    # parser.add_argument("--color", action="store_true", help="Use colormap to map the depth value to rgb channel")
    # parser.add_argument("--normalize", action="store_true", help="Normalize range map with LiDAR max range")

   
    return parser.parse_args()



def main(args):
    '''
    Assume the data is store in the following structure
    data
     - train
        - depth
            - all
                - image01
                - image02
                ...
        - intensity
            - all
                - image01
                - image02
                ...
     - val
        - depth
            ...
        - intensity
            ...
    '''
    dir_name = os.path.dirname(args.input_path)
    outputdir_fullpath = os.path.join(dir_name, args.output_path_name)

    # Create folder and directory
    outputpath = pathlib.Path(outputdir_fullpath)
    outputpath.parent.mkdir(parents = True, exist_ok = True)

    downsampling_factor = args.downsampling_factor

    for data_split in ['train', 'val']:
        output_image_dir = os.path.join(outputdir_fullpath, data_split)
        input_image_dir = os.path.join(args.input_path, data_split)

        for modality in ['depth', 'intensity']:
            images_path = os.path.join(os.path.join(input_image_dir, modality), 'all')

            outputdir_modality = os.path.join(os.path.join(output_image_dir, modality), 'all')
            outputdir_modality_fullpath = pathlib.Path(outputdir_modality)
            outputdir_modality_fullpath.parent.mkdir(parents = True, exist_ok = True)

            if os.path.exists(outputdir_modality):
                pass
            else:
                os.mkdir(outputdir_modality)


            images_fullpath = glob(images_path + "/*")
            for image_fullpath in images_fullpath:
                image_name = os.path.basename(image_fullpath)
                high_res_image = cv2.imread(image_fullpath, cv2.IMREAD_GRAYSCALE)
                h, _, = high_res_image.shape

                low_res_index = range(0, h, downsampling_factor)
                low_res_image = high_res_image[low_res_index, :]
                # print(low_res_image.shape)
                cv2.imwrite(os.path.join(outputdir_modality, image_name), low_res_image)

    print("Done")


if __name__ == "__main__":
    args = read_args()
    main(args)

import os
from PIL import Image
import numpy as np

# from torchvision import transforms
# from torchvision.datasets import ImageFolder
# import torch

# import torch.utils.data as data
# import torch.nn.functional as F

# from timm.data import create_transform
# from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
# from timm.data.dataset import ImageDataset
# import numpy as np

# import os
# import os.path
# import random
# from copy import deepcopy
# from typing import Any, Callable, Dict, List, Optional, Tuple, cast

# import numpy as np
# import torch
# from torchvision.datasets.vision import VisionDataset
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp', '.jpx')

image_rows_low = 32 
image_rows_high = 128 
image_cols = 2048
channel_num = 1
sensor_noise = 0.03
upscaling_factor = image_rows_high // image_rows_low

home_dir = "/cluster/work/riner/users/biyang/dataset/"
output_dir = "/cluster/work/riner/users/biyang/experiment/"
output_name = "LiDAR_Super_Resolution"
root_dir = os.path.join(output_dir, output_name)
# Check Path exists
path_lists = [root_dir]
for folder_name in path_lists:
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


training_data_path = os.path.join(home_dir, "depth_large_train.npy")
testing_data_path = os.path.join(home_dir, "depth_large_test.npy")


def pre_processing_raw_data(data_set_name, is_train = False):
    # load data
    # image_files = os.listdir(data_set_name)
    # full_res_data = np.array([np.array(Image.open(os.path.join(data_set_name, fname))).astype(np.float32) for fname in image_files])

    full_res_data = np.load(data_set_name)
    full_res_data = full_res_data.astype(np.float32, copy=True)
    full_res_data /= 255

    # add gaussian noise for [CARLA] data
    if is_train:
        print('add noise ...')
        noise = np.random.normal(0, sensor_noise, full_res_data.shape) # mu, sigma, size
        noise[full_res_data == 0] = 0
        full_res_data = full_res_data + noise
    return full_res_data


def get_low_res_from_high_res(high_res_data):
    low_res_index = range(0, image_rows_high, upscaling_factor)
    low_res_data = high_res_data[:,low_res_index]
    return low_res_data

def load_train_data():
    train_data = pre_processing_raw_data(training_data_path, is_train=True)
    train_data_input = get_low_res_from_high_res(train_data)
    return train_data_input, train_data

def load_test_data():
    test_data = pre_processing_raw_data(testing_data_path, is_train = False)
    test_data_input = get_low_res_from_high_res(test_data)
    return test_data_input, test_data




# For pytorch

# class ConcatDataset(torch.utils.data.Dataset):
#     def __init__(self, *datasets):
#         self.datasets = datasets

#     def __getitem__(self, i):
#         return tuple(d[i] for d in self.datasets)

#     def __len__(self):
#         return min(len(d) for d in self.datasets)


# def build_durlar_upsampling_dataset(is_train, args):
#     t_low_res = [transforms.Grayscale(), transforms.ToTensor()]
#     t_high_res = [transforms.Grayscale(), transforms.ToTensor()]
#     if args.crop:
#         t_low_res.append(transforms.CenterCrop(args.img_size_low_res))
#         t_high_res.append(transforms.CenterCrop(args.img_size_high_res))

#     transform_low_res = transforms.Compose(t_low_res)
#     transform_high_res = transforms.Compose(t_high_res)

#     root_low_res = os.path.join(args.data_path_low_res, 'train' if is_train else 'val')
#     root_low_res = os.path.join(root_low_res, 'depth')

#     root_high_res = os.path.join(args.data_path_high_res, 'train' if is_train else 'val')
#     root_high_res = os.path.join(root_high_res, 'depth')

#     dataset_low_res = ImageFolder(root_low_res, transform=transform_low_res)
#     dataset_high_res = ImageFolder(root_high_res, transform=transform_high_res)

#     assert len(dataset_high_res) == len(dataset_low_res)

#     dataset_concat = ConcatDataset(dataset_low_res, dataset_high_res)
#     return dataset_concat
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


training_data_path = os.path.join(home_dir, "depth_intensity_large/train/depth/all")
testing_data_path = os.path.join(home_dir, "depth_intensity_large/val/depth/all")

image_files_train = os.listdir(training_data_path)
full_res_data_train = np.array([np.array(Image.open(os.path.join(training_data_path, fname))).astype(np.float32) for fname in image_files_train])


image_files_test = os.listdir(testing_data_path)
full_res_data_test= np.array([np.array(Image.open(os.path.join(testing_data_path, fname))).astype(np.float32) for fname in image_files_test])

np.save(os.path.join(home_dir, "depth_large_train.npy"), full_res_data_train)
np.save(os.path.join(home_dir, "depth_large_test.npy"), full_res_data_test)



# def save_npy(data_set_name):
#     # load data
#     image_files = os.listdir(data_set_name)
#     full_res_data = np.array([np.array(Image.open(os.path.join(data_set_name, fname))).astype(np.float32) for fname in image_files])
    
#     np.save(os.path.join(home_dir, "depth_small_train.npy"), full_res_data)


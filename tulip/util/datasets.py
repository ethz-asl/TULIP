# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL
from PIL import Image

from torchvision import transforms
from torchvision.datasets import ImageFolder, DatasetFolder
import torch

import torch.utils.data as data
import torch.nn.functional as F

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data.dataset import ImageDataset
import numpy as np

import os
import os.path
import random
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import numpy as np
import torch
from torchvision.datasets.vision import VisionDataset
import copy
from pathlib import Path

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp', '.jpx')
NPY_EXTENSIONS = ('.npy', '.rimg', '.bin')
dataset_list = {}

def register_dataset(name):
    def decorator(cls):
        dataset_list[name] = cls
        return cls
    return decorator


def generate_dataset(args, is_train):
    dataset = dataset_list[args.dataset_select]
    return dataset(is_train, args)


class AddGaussianNoise(torch.nn.Module):
    def __init__(self, mu, sigma):
        super().__init__()#
        self.sigma = sigma
        self.mu = mu
    def __call__(self, img):
        return torch.randn(img.size()) * self.sigma + self.mu


    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size})"
    

class LogTransform(object):
    def __call__(self, tensor):
        return torch.log1p(tensor)


class CropRanges(object):
    def __init__(self, min_dist, max_dist):
        self.max_dist = max_dist
        self.min_dist = min_dist
    def __call__(self, tensor):
        mask = (tensor >= self.min_dist) & (tensor < self.max_dist)
        num_pixels = mask.sum()
        return torch.where(mask , tensor, 0), num_pixels

class KeepCloseScan(object):
    def __init__(self, max_dist):
        self.max_dist = max_dist
    def __call__(self, tensor):
        return torch.where(tensor < self.max_dist, tensor, 0)
    
class KeepFarScan(object):
    def __init__(self, min_dist):
        self.min_dist = min_dist
    def __call__(self, tensor):
        return torch.where(tensor > self.min_dist, tensor, 0)
    

class RandomRollRangeMap(object):
    """Roll Range Map along horizontal direction, 
    this requires the input and output have the same width 
    (downsampled only in vertical direction)"""
    def __init__(self, h_img = 2048, shift = None):
        if shift is not None:
            self.shift = shift
        else:
            self.shift = np.random.randint(0, h_img)
    def __call__(self, tensor):
        # Assume the dimension is B C H W
        return torch.roll(tensor, shifts = self.shift, dims = -1)

class DepthwiseConcatenation(object):
    """Concatenate the image depth wise -> one channel to multi-channels input"""
    
    def __init__(self, h_high_res: int, downsample_factor: int):
        self.low_res_indices = [range(i, h_high_res+i, downsample_factor) for i in range(downsample_factor)]

    def __call__(self, tensor):
        return torch.cat([tensor[:, self.low_res_indices[i], :] for i in range(len(self.low_res_indices))], dim = 0)

class DownsampleTensor(object):
    def __init__(self, h_high_res: int, downsample_factor: int, random = False):
        if random:
            index = np.random.randint(0, downsample_factor)
        else:
            index = 0
        self.low_res_index = range(0+index, h_high_res+index, downsample_factor)
    def __call__(self, tensor):
        return tensor[:, self.low_res_index, :]
    
class DownsampleTensorWidth(object):
    def __init__(self, w_high_res: int, downsample_factor: int, random = False):
        if random:
            index = np.random.randint(0, downsample_factor)
        else:
            index = 0
        self.low_res_index = range(0+index, w_high_res+index, downsample_factor)
    def __call__(self, tensor):
        return tensor[:, :, self.low_res_index]

class ScaleTensor(object):
    def __init__(self, scale_factor):
        self.scale_factor = scale_factor
    def __call__(self, tensor):
        return tensor*self.scale_factor
    
class FilterInvalidPixels(object):
    ''''Filter out pixels that are out of lidar range'''
    def __init__(self, min_range, max_range = 1):
        self.max_range = max_range
        self.min_range = min_range

    def __call__(self, tensor):
        return torch.where((tensor >= self.min_range) & (tensor <= self.max_range), tensor, 0)


class PairDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)
    

# def npy_loader(path: str) -> np.ndarray:
#     with open(path, "rb") as f:
#         range_map = np.load(f)
#     return range_map.astype(np.float32)

def bin_loader(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        range_intensity_map = np.fromfile(f, dtype=np.float32).reshape(64, 1024, 2)
        # range_map = range_intensity_map[..., 0]
    return range_intensity_map

def npy_loader(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        range_intensity_map = np.load(f)
        range_map = range_intensity_map[..., 0]
    return range_map.astype(np.float32)
    
def rimg_loader(path: str) -> np.ndarray:
    """
    Read range image from .rimg file (for CARLA dataset)
    """
    with open(path, 'rb') as f:
        size =  np.fromfile(f, dtype=np.uint, count=2)
        range_image = np.fromfile(f, dtype=np.float16)
    
    range_image = range_image.reshape(size[1], size[0])
    range_image = range_image.transpose()


    return np.flip(range_image).astype(np.float32)


class RangeMapFolder(DatasetFolder):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = npy_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        class_dir: bool = True,
    ):
        self.class_dir = class_dir
        super().__init__(
            root,
            loader,
            NPY_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )
        self.imgs = self.samples
        

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        if self.class_dir:
            return super().find_classes(directory)    
        else:
            return [""], {"":0}
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        name = os.path.basename(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return {'sample': sample,
                'class':target,
                'name': name}

@register_dataset('durlar')
def build_durlar_upsampling_dataset(is_train, args):
    input_size = tuple(args.img_size_low_res)
    output_size = tuple(args.img_size_high_res)

    t_low_res = [transforms.ToTensor(), ScaleTensor(1/120), FilterInvalidPixels(min_range = 0.3/120, max_range = 1)]
    t_high_res = [transforms.ToTensor(), ScaleTensor(1/120), FilterInvalidPixels(min_range = 0.3/120, max_range = 1)]

    t_low_res.append(DownsampleTensor(h_high_res=output_size[0], downsample_factor=output_size[0]//input_size[0]))

    if args.log_transform:
        t_low_res.append(LogTransform())
        t_high_res.append(LogTransform())
    
    if is_train and args.roll: 
        # t_low_res.append(AddGaussianNoise(sigma=0.03, mu=0))
        roll_low_res = RandomRollRangeMap()
        roll_high_res = RandomRollRangeMap(shift = roll_low_res.shift)
        t_low_res.append(roll_low_res)
        t_high_res.append(roll_high_res)

    transform_low_res = transforms.Compose(t_low_res)
    transform_high_res = transforms.Compose(t_high_res)

    root_low_res = os.path.join(args.data_path_low_res, 'train' if is_train else 'val')
    root_high_res = os.path.join(args.data_path_high_res, 'train' if is_train else 'val')

    dataset_low_res = RangeMapFolder(root_low_res, transform = transform_low_res, loader= npy_loader, class_dir=False)
    dataset_high_res = RangeMapFolder(root_high_res, transform = transform_high_res, loader =  npy_loader, class_dir = False)


    assert len(dataset_high_res) == len(dataset_low_res)

    dataset_concat = PairDataset(dataset_low_res, dataset_high_res)
    return dataset_concat

@register_dataset('kitti')
def build_kitti_upsampling_dataset(is_train, args):
    input_size = tuple(args.img_size_low_res)
    output_size = tuple(args.img_size_high_res)

    t_low_res = [transforms.ToTensor(), ScaleTensor(1/80)]
    t_high_res = [transforms.ToTensor(), ScaleTensor(1/80)]

    t_low_res.append(DownsampleTensor(h_high_res=output_size[0], downsample_factor=output_size[0]//input_size[0],))
    if output_size[1] // input_size[1] > 1:
        t_low_res.append(DownsampleTensorWidth(w_high_res=output_size[1], downsample_factor=output_size[1]//input_size[1],))

    if args.log_transform:
        t_low_res.append(LogTransform())
        t_high_res.append(LogTransform())

    transform_low_res = transforms.Compose(t_low_res)
    transform_high_res = transforms.Compose(t_high_res)        

    root_low_res = os.path.join(args.data_path_low_res, 'train' if is_train else 'val')
    root_high_res = os.path.join(args.data_path_high_res, 'train' if is_train else 'val')


    dataset_low_res = RangeMapFolder(root_low_res, transform = transform_low_res, loader= npy_loader, class_dir = False)
    dataset_high_res = RangeMapFolder(root_high_res, transform = transform_high_res, loader = npy_loader, class_dir = False)

    assert len(dataset_high_res) == len(dataset_low_res)

    dataset_concat = PairDataset(dataset_low_res, dataset_high_res)
    return dataset_concat
    

@register_dataset('carla')
def build_carla_upsampling_dataset(is_train, args):
    # Carla dataset is not normalized
    input_size = tuple(args.img_size_low_res)
    output_size = tuple(args.img_size_high_res)
    input_img_path = str(input_size[0]) + '_' + str(input_size[1])
    output_img_path = str(output_size[0]) + '_' + str(output_size[1])

    available_resolution = os.listdir(os.path.join(args.data_path_low_res, 'Town01'))

    t_low_res = [transforms.ToTensor(), ScaleTensor(1/80), FilterInvalidPixels(min_range = 2/80, max_range = 1)]
    t_high_res = [transforms.ToTensor(), ScaleTensor(1/80), FilterInvalidPixels(min_range = 2/80, max_range = 1)]


    INPUT_DATA_UNAVAILABLE = input_img_path not in available_resolution and output_img_path in available_resolution

    if INPUT_DATA_UNAVAILABLE:
        print("There is no data for the specified input size but output size is available, Downsample input data from the output")
        t_low_res.append(DownsampleTensor(h_high_res=output_size[0], downsample_factor=output_size[0]//input_size[0], ))

    if args.log_transform:
        t_low_res.append(LogTransform())
        t_high_res.append(LogTransform())

    transform_low_res = transforms.Compose(t_low_res)
    transform_high_res = transforms.Compose(t_high_res)

    scene_ids = ['Town01',
                 'Town02',
                 'Town03',
                 'Town04',
                 'Town05',
                 'Town06',] if is_train else ['Town07', 'Town10HD']

    scenes_data_input = []
    scenes_data_output = []
    
    for scene_ids_i in scene_ids:
        if INPUT_DATA_UNAVAILABLE:
            input_scene_datapath = os.path.join(args.data_path_low_res, scene_ids_i, output_img_path)
            output_scene_datapath = os.path.join(args.data_path_high_res, scene_ids_i, output_img_path)
            scenes_data_input.append(RangeMapFolder(input_scene_datapath, transform = transform_low_res, loader=rimg_loader, class_dir=False))
            scenes_data_output.append(RangeMapFolder(output_scene_datapath, transform = transform_high_res, loader=rimg_loader, class_dir=False))

        else:

            input_scene_datapath = os.path.join(args.data_path_low_res, scene_ids_i, input_img_path)
            output_scene_datapath = os.path.join(args.data_path_high_res, scene_ids_i, output_img_path)
            scenes_data_input.append(RangeMapFolder(input_scene_datapath, transform = transform_low_res, loader=rimg_loader, class_dir=False))
            scenes_data_output.append(RangeMapFolder(output_scene_datapath, transform = transform_high_res, loader=rimg_loader, class_dir=False))

    
    input_data = data.ConcatDataset(scenes_data_input)
    output_data = data.ConcatDataset(scenes_data_output)

    carla_dataset = PairDataset(input_data, output_data)

    return carla_dataset


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

from torchvision import datasets, transforms
import torch

import torch.utils.data as data

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
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp', '.jpx')

def build_durlar_dataset(is_train, args):
    
    t = [transforms.Grayscale(), transforms.ToTensor()]
    if args.crop:
        if is_train:
            # t.append(transforms.RandomCrop(args.img_size))
            t.append(transforms.CenterCrop(args.img_size))
        else:
            t.append(transforms.CenterCrop(args.img_size))

    transform = transforms.Compose(t)
    root = os.path.join(args.data_path, 'train' if is_train else 'val')

    root = os.path.join(root, 'depth')
    dataset = ImageDataset(root, transform=transform)

    return dataset

    
def build_dataset(is_train, args):
    # if args.transform:
    transform = build_transform(is_train, args)
    if args.gray_scale:
        transform = transforms.Compose([transform, transforms.Grayscale(num_output_channels=1)])

    root = os.path.join(args.data_path, 'train' if is_train else 'val')
    dataset = ImageDataset(root, transform=transform)

    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    
    # t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


class ComposeDepthIntensity(object):
    def __init__(self, args, is_train = False):
        self.is_train = is_train

    def __call__(self, task_dict):
        # Convert to Tensor
        depth_img = task_dict['depth']
        intensity_img = task_dict['intensity']

        depth_img = torch.Tensor(np.array(depth_img)).unsqueeze(0)
        intensity_img = torch.Tensor(np.array(intensity_img)).unsqueeze(0) # 1xHxW

        depth_img = transforms.functional.center_crop(depth_img, (128, 128))
        intensity_img = transforms.functional.center_crop(intensity_img, (128, 128))

        depth_intensity = torch.cat([depth_img, intensity_img], dim=0) 
        
        return depth_intensity

    def __repr__(self):
        repr = "(DataAugmentationForMultiMAE,\n"
        #repr += "  transform = %s,\n" % str(self.transform)
        repr += ")"
        return repr

def build_depth_intensity_dataset(args, is_train = True):
    # transform = DataAugmentationForMultiMAE(args)
    # hard_coded for DurLAR
    # transform = transforms.Compose([transforms.ToTensor(), transforms.Grayscale()]) 
    transform = ComposeDepthIntensity(args, is_train=is_train)
    if is_train:
        return MultiTaskImageFolder(os.path.join(args.data_path, "train"), ['depth', 'intensity'], transform=transform)
    else:
        return MultiTaskImageFolder(os.path.join(args.data_path, "val"), ['depth', 'intensity'], transform=transform)
    
def pil_loader(path: str, convert_rgb=True) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    # with open(path, 'rb') as f:
    #     img = Image.open(f)
    img = Image.open(path)
    return img.convert('RGB') if convert_rgb else img



class MultiTaskDatasetFolder(VisionDataset):
    """A generic multi-task dataset loader where the samples are arranged in this way: ::

        root/task_a/class_x/xxx.ext
        root/task_a/class_y/xxy.ext
        root/task_a/class_z/xxz.ext

        root/task_b/class_x/xxx.ext
        root/task_b/class_y/xxy.ext
        root/task_b/class_z/xxz.ext

    Args:
        root (string): Root directory path.
        tasks (list): List of tasks as strings
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt logs)
            both extensions and is_valid_file should not be passed.

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(
            self,
            root: str,
            tasks: List[str],
            loader: Callable[[str], Any],
            extensions: Optional[Tuple[str, ...]] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            prefixes: Optional[Dict[str,str]] = None,
            max_images: Optional[int] = None
    ) -> None:
        super(MultiTaskDatasetFolder, self).__init__(root, transform=transform,
                                            target_transform=target_transform)
        self.tasks = tasks
        classes, class_to_idx = self._find_classes(os.path.join(self.root, self.tasks[0]))

        prefixes = {} if prefixes is None else prefixes
        prefixes.update({task: '' for task in tasks if task not in prefixes})
        
        samples = {
            task: make_dataset(os.path.join(self.root, f'{prefixes[task]}{task}'), class_to_idx, extensions, is_valid_file)
            for task in self.tasks
        }
        
        for task, task_samples in samples.items():
            if len(task_samples) == 0:
                msg = "Found 0 logs in subfolders of: {}\n".format(os.path.join(self.root, task))
                if extensions is not None:
                    msg += "Supported extensions are: {}".format(",".join(extensions))
                raise RuntimeError(msg)

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        # self.targets = [s[1] for s in list(samples.values())[0]]

        # Select random subset of dataset if so specified
        if isinstance(max_images, int):
            total_samples = len(list(self.samples.values())[0])
            np.random.seed(0)
            permutation = np.random.permutation(total_samples)
            for task in samples:
                self.samples[task] = [self.samples[task][i] for i in permutation][:max_images]
        
        self.cache = {}

    def _find_classes(self, dir: str) -> Tuple[List[str], Dict[str, int]]:
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        if index in self.cache:
            sample_dict, target = deepcopy(self.cache[index])
        else:
            sample_dict = {}
            for task in self.tasks:
                path, target = self.samples[task][index]
                sample = pil_loader(path, convert_rgb=(task=='rgb'))
                sample = sample.convert('P') if 'semseg' in task else sample
                sample_dict[task] = sample
            # self.cache[index] = deepcopy((sample_dict, target))

        if self.transform is not None:
            sample_dict = self.transform(sample_dict)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample_dict, target

    def __len__(self) -> int:
        return len(list(self.samples.values())[0])
    

def has_file_allowed_extension(filename: str, extensions: Tuple[str, ...]) -> bool:
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def is_image_file(filename: str) -> bool:
    """Checks if a file is an allowed image extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def make_dataset(
        directory: str,
        class_to_idx: Dict[str, int],
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
) -> List[Tuple[str, int]]:
    instances = []
    directory = os.path.expanduser(directory)
    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x: str) -> bool:
            return has_file_allowed_extension(x, cast(Tuple[str, ...], extensions))
    is_valid_file = cast(Callable[[str], bool], is_valid_file)
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = path, class_index
                    instances.append(item)
    return instances


class MultiTaskImageFolder(MultiTaskDatasetFolder):
    """A generic multi-task dataset loader where the images are arranged in this way: ::

        root/task_a/class_x/xxx.ext
        root/task_a/class_y/xxy.ext
        root/task_a/class_z/xxz.ext

        root/task_b/class_x/xxx.ext
        root/task_b/class_y/xxy.ext
        root/task_b/class_z/xxz.ext

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt logs)

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(
            self,
            root: str,
            tasks: List[str],
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = pil_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            prefixes: Optional[Dict[str,str]] = None,
            max_images: Optional[int] = None
    ):
        super(MultiTaskImageFolder, self).__init__(root, tasks, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file,
                                          prefixes=prefixes,
                                          max_images=max_images)
        self.imgs = self.samples
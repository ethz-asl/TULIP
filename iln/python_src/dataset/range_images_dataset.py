import torch
from torch.utils.data import Dataset

import numpy as np
import os

from dataset.dataset_utils import *

@register_dataset('range_images_kitti')
class KittiDataset(Dataset):
    def __init__(self, directory, high_res_path, low_res_path, res_out = (64, 1024), res_in = (16, 1024)):
        """
        Constructor of dataset class (pair: input range image & output range image).

        :param directory: directory of dataset
        :param scene_ids: scene IDs of dataset
        :param res_in: input resolution of range image
        :param res_out: output resolution of range image
        :param memory_fetch: on/off for fetching all the data into memory storage
        """
        super(KittiDataset, self).__init__()

        # Dataset configurations
        self.dataset_directory = directory

        # Read the LiDAR configurations
        lidar_config_filename = os.path.join(directory, 'kitti.yaml')
        self.lidar_in = initialize_lidar(lidar_config_filename, channels=int(res_in[0]), points_per_ring=int(res_in[1]))
        self.lidar_out = initialize_lidar(lidar_config_filename, channels=int(res_out[0]), points_per_ring=int(res_out[1]))

        # Read all the filenames
        # self.input_range_image_filenames = []
        # self.output_range_image_filenames = []


        self.input_filenames = [os.path.join(low_res_path, f) for f in os.listdir(low_res_path) if f.endswith('.npy')]
        self.input_filenames.sort()


        self.output_filenames = [os.path.join(high_res_path, f) for f in os.listdir(high_res_path) if f.endswith('.npy')]
        self.output_filenames.sort()

        assert (len(self.input_filenames) == len(self.output_filenames))

        # self.input_range_image_filenames.extend(input_filenames)
        # self.output_range_image_filenames.extend(output_filenames)


    def __len__(self):
        """
        Get the number of range image pairs (dataset size).

        :return dataset size
        """
        return len(self.output_filenames)

    def __getitem__(self, item):
        """
        Get a pair of input and output range images assigned to an index.

        :param item: index of pair
        :return normalized range image pair
        """

        # Read the pair of input and output range images
        input_range_image_filename = self.input_filenames[item]
        output_range_image_filename = self.output_filenames[item]
        input_range_image = read_range_kitti(input_range_image_filename)
        
        # Downsample from the high res image
        input_range_image = downsample_range_durlar(input_range_image, 
                                                    h_high_res=self.lidar_out['channels'], 
                                                    downsample_factor=self.lidar_out['channels'] // self.lidar_in['channels'])

        output_range_image = read_range_kitti(output_range_image_filename)

        output_range_image = output_range_image / self.lidar_out['norm_r']
        input_range_image = input_range_image / self.lidar_in['norm_r']

        # Normalization ([0, 1] -> [-1, 1])
        input_range_image *= 2.0
        input_range_image -= 1.0
        output_range_image *= 2.0
        output_range_image -= 1.0

        return input_range_image[np.newaxis, :, :], output_range_image[np.newaxis, :, :]


@register_dataset('range_images_durlar')
class DurlarDataset(Dataset):
    def __init__(self, directory, high_res_path, low_res_path, res_out = (128, 2048), res_in = (32, 2048)):
        """
        Constructor of dataset class (pair: input range image & output range image).

        :param directory: directory of dataset
        :param scene_ids: scene IDs of dataset
        :param res_in: input resolution of range image
        :param res_out: output resolution of range image
        :param memory_fetch: on/off for fetching all the data into memory storage
        """
        super(DurlarDataset, self).__init__()

        # Dataset configurations
        self.dataset_directory = directory

        # Read the LiDAR configurations
        lidar_config_filename = os.path.join(directory, 'durlar.yaml')
        self.lidar_in = initialize_lidar(lidar_config_filename, channels=int(res_in[0]), points_per_ring=int(res_in[1]))
        self.lidar_out = initialize_lidar(lidar_config_filename, channels=int(res_out[0]), points_per_ring=int(res_out[1]))

        # Read all the filenames
        # self.input_range_image_filenames = []
        # self.output_range_image_filenames = []


        self.input_filenames = [os.path.join(low_res_path, f) for f in os.listdir(low_res_path) if f.endswith('.npy')]
        self.input_filenames.sort()


        self.output_filenames = [os.path.join(high_res_path, f) for f in os.listdir(high_res_path) if f.endswith('.npy')]
        self.output_filenames.sort()

        assert (len(self.input_filenames) == len(self.output_filenames))

        # self.input_range_image_filenames.extend(input_filenames)
        # self.output_range_image_filenames.extend(output_filenames)


    def __len__(self):
        """
        Get the number of range image pairs (dataset size).

        :return dataset size
        """
        return len(self.output_filenames)

    def __getitem__(self, item):
        """
        Get a pair of input and output range images assigned to an index.

        :param item: index of pair
        :return normalized range image pair
        """

        # Read the pair of input and output range images
        input_range_image_filename = self.input_filenames[item]
        output_range_image_filename = self.output_filenames[item]
        input_range_image = read_range_durlar(input_range_image_filename)
        
        # Downsample from the high res image
        input_range_image = downsample_range_durlar(input_range_image, 
                                                    h_high_res=self.lidar_out['channels'], 
                                                    downsample_factor=self.lidar_out['channels'] // self.lidar_in['channels'])

        output_range_image = read_range_durlar(output_range_image_filename)

        # Crop the values out of the detection range
        input_range_image[input_range_image < 10e-10] = self.lidar_in['norm_r']
        input_range_image[input_range_image < self.lidar_in['min_r'] / self.lidar_in['max_r']] = 0.0
        input_range_image[input_range_image > self.lidar_in['max_r'] / self.lidar_in['max_r']] = self.lidar_in['norm_r']
        output_range_image[output_range_image < 10e-10] = self.lidar_out['norm_r']
        output_range_image[output_range_image < self.lidar_out['min_r'] / self.lidar_in['max_r']] = 0.0
        output_range_image[output_range_image > self.lidar_out['max_r'] / self.lidar_in['max_r']] = self.lidar_out['norm_r']


        # Normalization ([0, 1] -> [-1, 1])
        input_range_image *= 2.0
        input_range_image -= 1.0
        output_range_image *= 2.0
        output_range_image -= 1.0

        return input_range_image[np.newaxis, :, :], output_range_image[np.newaxis, :, :]




@register_dataset('range_images')
class RangeImagesDataset(Dataset):
    def __init__(self, directory, scene_ids, res_in, res_out, memory_fetch=True):
        """
        Constructor of dataset class (pair: input range image & output range image).

        :param directory: directory of dataset
        :param scene_ids: scene IDs of dataset
        :param res_in: input resolution of range image
        :param res_out: output resolution of range image
        :param memory_fetch: on/off for fetching all the data into memory storage
        """
        super(RangeImagesDataset, self).__init__()

        # Dataset configurations
        self.dataset_directory = directory
        self.scene_ids = scene_ids
        self.res_in = res_in
        self.res_out = res_out
        self.memory_fetch = memory_fetch

        # Read the LiDAR configurations
        lidar_config_filename = os.path.join(directory, 'lidar_specification.yaml')
        self.lidar_in = initialize_lidar(lidar_config_filename, channels=int(res_in.split('_')[0]), points_per_ring=int(res_in.split('_')[1]))
        self.lidar_out = initialize_lidar(lidar_config_filename, channels=int(res_out.split('_')[0]), points_per_ring=int(res_out.split('_')[1]))

        # Read all the filenames
        self.input_range_image_filenames = []
        self.output_range_image_filenames = []


        availbale_data = os.listdir(os.path.join(directory, scene_ids[0], res_out))
        self.INPUT_AVAILABLE = res_in in availbale_data

        for scene_id in scene_ids:
            

            output_directory = os.path.join(directory, scene_id, res_out)
            output_filenames = [os.path.join(output_directory, f) for f in os.listdir(output_directory) if f.endswith('.rimg')]
            output_filenames.sort()

            if self.INPUT_AVAILABLE:
                input_directory = os.path.join(directory, scene_id, res_in)
                input_filenames = [os.path.join(input_directory, f) for f in os.listdir(input_directory) if f.endswith('.rimg')]
                input_filenames.sort()
            else:
                input_filenames = output_filenames
            assert (len(input_filenames) == len(output_filenames))

            self.input_range_image_filenames.extend(input_filenames)
            self.output_range_image_filenames.extend(output_filenames)

        # Fetch all the data pairs into the memory storage
        # NOTE: This can provide fast training/testing despite requiring a large memory size
        if memory_fetch:
            num_of_data_per_pair = self.lidar_in['channels'] * self.lidar_in['points_per_ring'] + self.lidar_out['channels'] * self.lidar_out['points_per_ring']
            self.range_image_pairs = np.zeros((len(self.input_range_image_filenames), num_of_data_per_pair), dtype=np.float32)
            self.pair_split_idx = self.lidar_in['channels'] * self.lidar_in['points_per_ring']

            for idx, filenames in enumerate(zip(self.input_range_image_filenames, self.output_range_image_filenames)):
                self.range_image_pairs[idx, :self.pair_split_idx] = read_range_image_binary(filenames[0]).flatten()
                self.range_image_pairs[idx, self.pair_split_idx:] = read_range_image_binary(filenames[1]).flatten()

            # Crop the values out of the detection range
            self.range_image_pairs[self.range_image_pairs < 10e-10] = self.lidar_out['norm_r']
            self.range_image_pairs[self.range_image_pairs < self.lidar_out['min_r']] = 0.0
            self.range_image_pairs[self.range_image_pairs > self.lidar_out['max_r']] = self.lidar_out['norm_r']

            # Normalization (in-place operations instead of function call)
            self.range_image_pairs *= (2.0 / self.lidar_out['norm_r'])
            self.range_image_pairs -= 1.0

    def __len__(self):
        """
        Get the number of range image pairs (dataset size).

        :return dataset size
        """
        return len(self.output_range_image_filenames)

    def __getitem__(self, item):
        """
        Get a pair of input and output range images assigned to an index.

        :param item: index of pair
        :return normalized range image pair
        """
        if self.memory_fetch:
            # Read the pair of input and output range images (normalized)
            input_range_image = self.range_image_pairs[item, :self.pair_split_idx].reshape(self.lidar_in['channels'], self.lidar_in['points_per_ring'])
            output_range_image = self.range_image_pairs[item, self.pair_split_idx:].reshape(self.lidar_out['channels'], self.lidar_out['points_per_ring'])

        else:
            # Read the pair of input and output range images
            input_range_image_filename = self.input_range_image_filenames[item]
            output_range_image_filename = self.output_range_image_filenames[item]

            if self.INPUT_AVAILABLE:
                input_range_image = read_range_image_binary(input_range_image_filename)
            else:
                input_range_image = read_and_downsample_range_image_binary(input_range_image_filename, 
                                                                           downsample_factor=int(self.res_out.split('_')[0]) // int(self.res_in.split('_')[0]))
            output_range_image = read_range_image_binary(output_range_image_filename)

            # Crop the values out of the detection range
            input_range_image[input_range_image < 10e-10] = self.lidar_in['norm_r']
            input_range_image[input_range_image < self.lidar_in['min_r']] = 0.0
            input_range_image[input_range_image > self.lidar_in['max_r']] = self.lidar_in['norm_r']
            output_range_image[output_range_image < 10e-10] = self.lidar_out['norm_r']
            output_range_image[output_range_image < self.lidar_out['min_r']] = 0.0
            output_range_image[output_range_image > self.lidar_out['max_r']] = self.lidar_out['norm_r']

            # Normalization
            input_range_image *= (2.0 / self.lidar_in['norm_r'])
            input_range_image -= 1.0
            output_range_image *= (2.0 / self.lidar_out['norm_r'])
            output_range_image -= 1.0

        return input_range_image[np.newaxis, :, :], output_range_image[np.newaxis, :, :]

    def get_range_image_pair(self, map_id, scan_number):
        """
        Get a pair of input and output range images assigned to the same scan number in a map.

        :param map_id: town id
        :param scan_number: scan number
        :return: normalized range image pair
        """
        # Find the item index associated with the scan number
        output_range_image_filename = os.path.join(self.dataset_directory, map_id, self.res_out, str(scan_number) + '.rimg')
        item_idx = self.output_range_image_filenames.index(output_range_image_filename)
        return self[item_idx]

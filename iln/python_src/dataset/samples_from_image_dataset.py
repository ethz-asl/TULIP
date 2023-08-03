import torch
from torch.utils.data import Dataset

import numpy as np
import os

from dataset.dataset_utils import register_dataset, normalization_queries
from dataset.range_images_dataset import RangeImagesDataset, DurlarDataset



@register_dataset('range_samples_from_durlar')
class SamplesFromDurlarDataset(DurlarDataset):
    def __init__(self, directory, high_res_path, low_res_path, res_out = (128, 2048), res_in = (32, 2048), num_of_samples=0):
        """
        Constructor of dataset class (pair: input range image & output range samples)

        :param directory: directory of dataset
        :param scene_ids: scene IDs of dataset
        :param res_in: input resolution of range image
        :param res_out: output resolution of range image
        :param num_of_samples: the number of samples used for training/testing (0: the use of all the samples)
        :param memory_fetch: on/off for fetching all the data into memory storage
        """
        super(SamplesFromDurlarDataset, self).__init__(directory, high_res_path, low_res_path, res_out, res_in)

        # Dataset configurations
        self.num_of_samples = num_of_samples

        # Pre-compute the query laser directions of output range samples
        v_dir = np.linspace(start=self.lidar_out['min_v'], stop=self.lidar_out['max_v'], num=self.lidar_out['channels'])
        h_dir = np.linspace(start=self.lidar_out['min_h'], stop=self.lidar_out['max_h'], num=self.lidar_out['points_per_ring'], endpoint=False)

        v_angles = []
        h_angles = []

        for i in range(self.lidar_out['channels']):
            v_angles = np.append(v_angles, np.ones(self.lidar_out['points_per_ring']) * v_dir[i])
            h_angles = np.append(h_angles, h_dir)

        self.queries = np.stack((v_angles, h_angles), axis=-1).astype(np.float32)
        self.queries = normalization_queries(self.queries, self.lidar_in)

    def __len__(self):
        """
        Get the number of range image pairs (dataset size)

        :return dataset size
        """
        return len(self.output_filenames)

    def __getitem__(self, item):
        """
        Get a pair of input range image and output range samples assigned to an index

        :param item: index of pair
        :return normalized range data pair (output data can be sub-sampled)
        """
        # Read the normalized range image pair
        input_range_image, output_range_image = super().__getitem__(item)

        max_num_of_samples = output_range_image.shape[1] * output_range_image.shape[2]
        if 0 < self.num_of_samples < max_num_of_samples:
            # Sub-sample the query laser directions (queries) and their values
            sample_idx = np.random.choice(max_num_of_samples, self.num_of_samples, replace=False)
            output_queries = self.queries[sample_idx]
            output_ranges = output_range_image.flatten()[sample_idx]
            return input_range_image, output_queries, output_ranges[:, np.newaxis]
        else:
            # Use all the output samples without sub-sampling
            output_ranges = output_range_image.flatten()
            return input_range_image, self.queries, output_ranges[:, np.newaxis]
        




@register_dataset('range_samples_from_image')
class SamplesFromImageDataset(RangeImagesDataset):
    def __init__(self, directory, scene_ids, res_in, res_out, num_of_samples=0, memory_fetch=True):
        """
        Constructor of dataset class (pair: input range image & output range samples)

        :param directory: directory of dataset
        :param scene_ids: scene IDs of dataset
        :param res_in: input resolution of range image
        :param res_out: output resolution of range image
        :param num_of_samples: the number of samples used for training/testing (0: the use of all the samples)
        :param memory_fetch: on/off for fetching all the data into memory storage
        """
        super(SamplesFromImageDataset, self).__init__(directory, scene_ids, res_in, res_out, memory_fetch)

        # Dataset configurations
        self.num_of_samples = num_of_samples

        # Pre-compute the query laser directions of output range samples
        v_dir = np.linspace(start=self.lidar_out['min_v'], stop=self.lidar_out['max_v'], num=self.lidar_out['channels'])
        h_dir = np.linspace(start=self.lidar_out['min_h'], stop=self.lidar_out['max_h'], num=self.lidar_out['points_per_ring'], endpoint=False)

        v_angles = []
        h_angles = []

        for i in range(self.lidar_out['channels']):
            v_angles = np.append(v_angles, np.ones(self.lidar_out['points_per_ring']) * v_dir[i])
            h_angles = np.append(h_angles, h_dir)

        self.queries = np.stack((v_angles, h_angles), axis=-1).astype(np.float32)
        self.queries = normalization_queries(self.queries, self.lidar_in)

    def __len__(self):
        """
        Get the number of range image pairs (dataset size)

        :return dataset size
        """
        return len(self.output_range_image_filenames)

    def __getitem__(self, item):
        """
        Get a pair of input range image and output range samples assigned to an index

        :param item: index of pair
        :return normalized range data pair (output data can be sub-sampled)
        """
        # Read the normalized range image pair
        input_range_image, output_range_image = super().__getitem__(item)

        max_num_of_samples = output_range_image.shape[1] * output_range_image.shape[2]
        if 0 < self.num_of_samples < max_num_of_samples:
            # Sub-sample the query laser directions (queries) and their values
            sample_idx = np.random.choice(max_num_of_samples, self.num_of_samples, replace=False)
            output_queries = self.queries[sample_idx]
            output_ranges = output_range_image.flatten()[sample_idx]
            return input_range_image, output_queries, output_ranges[:, np.newaxis]
        else:
            # Use all the output samples without sub-sampling
            output_ranges = output_range_image.flatten()
            return input_range_image, self.queries, output_ranges[:, np.newaxis]

    def get_range_samples_pair(self, map_id, scan_number):
        """
        Get a pair of input range image and output range samples assigned to the same scan number in a map

        :param map_id: town id
        :param scan_number: scan number
        :return: the normalized pair of range image and range samples
        """
        output_range_image_filename = os.path.join(self.dataset_directory, map_id, self.res_out, str(scan_number) + '.rimg')
        item_idx = self.output_range_image_filenames.index(output_range_image_filename)
        return self[item_idx]

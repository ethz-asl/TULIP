#!/usr/bin/env python3

import argparse
import copy

import torch
import torch.nn as nn

# Datasets
from dataset.dataset_utils import points_to_range_image, range_image_to_points, generate_laser_directions, initialize_lidar
from dataset.dataset_utils import normalization_queries

# Models
from models.iln.iln import ILN
from models.liif_cvpr21.liif_lidar import LIIFLiDAR
from models.interpolation.interpolation import Interpolation
from models.model_utils import generate_model

# ROS
import rospy
import ros_numpy
from sensor_msgs.msg import PointCloud2
from visualization.visualization_utils import get_pointcloud_xyz

import numpy as np


class LiDARSuperResolution:
    def __init__(self, checkpoint, target_resolution='64_1024', lidar_in=None):
        """
        Constructor

        :param checkpoint: check point filename [.pth]
        :param target_resolution: vertical and horizontal target LiDAR resolution (default: 64_1024)
        :param lidar_in: lidar configuration (default: configuration loaded from the check point file)
        """
        # ROS node settings
        rospy.init_node('demo_lidar_super_resolution')
        rospy.Subscriber('/pointcloud_in', PointCloud2, callback=self.callback, queue_size=1)
        self.pointcloud_out_publisher = rospy.Publisher('/pointcloud_out', PointCloud2, queue_size=1)

        # Load the check point
        config = torch.load(checkpoint)

        # Model
        self.network = generate_model(config['model']['name'], config['model']['args'])
        self.network.load_state_dict(config['model']['state_dict'])
        if torch.cuda.device_count() > 1:
            model = nn.parallel.DataParallel(self.network)
        self.network.eval().cuda()

        # Prepare the trained LiDAR specification
        self.lidar_in = lidar_in if lidar_in is not None else config['lidar_in']
        self.lidar_out = copy.deepcopy(self.lidar_in)
        self.lidar_out['channels'] = int(target_resolution.split('_')[0])
        self.lidar_out['points_per_ring'] = int(target_resolution.split('_')[1])

        # Pre-compute the query lasers (normalized)
        self.query_lasers = generate_laser_directions(self.lidar_out)
        self.query_lasers = normalization_queries(self.query_lasers, self.lidar_in)
        self.query_lasers = torch.from_numpy(self.query_lasers)[None, :, :].cuda()

        self.pred_batch = 100000

        # Logs
        rospy.loginfo('Loaded model: %s', config['model']['name'])
        rospy.loginfo('Target resolution: %d x %d', self.lidar_out['channels'], self.lidar_out['points_per_ring'])

    def callback(self, input_pointcloud2_msg):
        """
        Generate high-resolution point cloud (prediction) when the low-resolution point cloud is captured by sensor.

        :param input_pointcloud2_msg: input point cloud with ROS PointCloud2 type
        """
        # Convert the input points into the range image (normalized)
        input_points = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(input_pointcloud2_msg).astype(np.float32)
        input_range_image = points_to_range_image(input_points, lidar=self.lidar_in)

        # Inplace normalization without 'normalization_ranges()' function call
        input_range_image = torch.from_numpy(input_range_image)[None, None, :, :].cuda()
        input_range_image *= (2.0 / self.lidar_in['norm_r'])
        input_range_image -= 1.0

        # Reconstruct the up-scaled output range image (normalized)
        with torch.no_grad():
            self.network.gen_feat(input_range_image)
            n = self.query_lasers.shape[1]
            ql = 0
            preds = []
            while ql < n:
                qr = min(ql + self.pred_batch, n)
                pred = self.network.query_detection(self.query_lasers[:, ql:qr, :])
                preds.append(pred)
                ql = qr

        pred_range_image = torch.cat(preds, dim=1).view(-1).reshape(self.lidar_out['channels'], self.lidar_out['points_per_ring'])

        # Inplace denormalization without 'denormalization_ranges()' function call
        pred_range_image += 1.0
        pred_range_image *= (0.5 * self.lidar_out['norm_r'])
        # Remove the values out of detection ranges
        pred_range_image[pred_range_image < self.lidar_out['min_r']] = 0.0
        pred_range_image[pred_range_image > self.lidar_out['max_r']] = 0.0

        pred_range_image = pred_range_image.cpu().numpy()

        # Convert the predicted range image into the output points
        # TODO: Add noise filters; e.g., PCL noise SOR(statistical outlier removal) filter
        output_points = range_image_to_points(range_image=pred_range_image, lidar=self.lidar_out)

        # Publish the output
        if self.pointcloud_out_publisher.get_num_connections() > 0:
            output_pointcloud_msg = get_pointcloud_xyz(points=output_points, stamp=rospy.Time.now(), frame_id=input_pointcloud2_msg.header.frame_id)
            self.pointcloud_out_publisher.publish(output_pointcloud_msg)


if __name__ == '__main__':
    # Parse the arguments
    parser = argparse.ArgumentParser(description="Demonstrate the LiDAR points reconstruction in real-time")
    parser.add_argument('-cp', '--checkpoint',
                        type=str,
                        required=True,
                        help='Check point filename. [.pth]')
    parser.add_argument('-r', '--target_resolution',
                        type=str,
                        required=False,
                        default='64_1024',
                        help='Vertical and horizontal target resolution; (default: 64_1024)')
    parser.add_argument('-l', '--lidar',
                        type=str,
                        required=False,
                        default=None,
                        help='LiDAR specification (default: the spec used in network training)')
    args, unknown = parser.parse_known_args()

    # ROS node construction
    lidar_in = initialize_lidar(args.lidar, 16, 1024) if args.lidar is not None else None
    lidar_super_resolution_node = LiDARSuperResolution(checkpoint=args.checkpoint,
                                                       target_resolution=args.target_resolution,
                                                       lidar_in=lidar_in)

    rospy.spin()

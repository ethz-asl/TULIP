import argparse
import copy
import time

import torch
import torch.nn as nn
import numpy as np

# Datasets
from dataset.dataset_utils import read_range_image_binary, generate_laser_directions, range_image_to_points
from dataset.dataset_utils import normalization_ranges, denormalization_ranges, normalization_queries

# Models
from models.iln.iln import ILN
from models.liif_cvpr21.liif_lidar import LIIFLiDAR
from models.interpolation.interpolation import Interpolation
from models.model_utils import generate_model

# ROS for visualization
import rospy
from sensor_msgs.msg import PointCloud2
from visualization.visualization_utils import get_pointcloud_xyz

import voxelizer


def predict_detection_distances(input_image, pixel_centers, pred_batch=100000):
    """
    Predict a high-resolution range image (prediction) from low-resolution image (input) and pixel centers (queries).

    :param input_image: low-resolution LiDAR range image
    :param pixel_centers: query lasers associated with pixel centers of range image
    :param pred_batch: batch size for predicting the detection distances (default: 100000)
    :return high-resolution LiDAR range image
    """
    input_image = torch.from_numpy(input_image)[None, None, :, :].cuda()
    pixel_centers = torch.from_numpy(pixel_centers)[None, :, :].cuda()

    with torch.no_grad():
        model.gen_feat(input_image)
        n = pixel_centers.shape[1]
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + pred_batch, n)
            pred = model.query_detection(pixel_centers[:, ql:qr, :])
            preds.append(pred)
            ql = qr

    return torch.cat(preds, dim=1).view(-1).cpu().numpy()


if __name__ == '__main__':
    # Parse the arguments
    parser = argparse.ArgumentParser(description="Demonstrate the LiDAR points reconstruction to various target resolutions")
    parser.add_argument('-i', '--input_filename',
                        type=str,
                        required=True,
                        help='Input range image [.rimg]; e.g.) Carla/Town07/16_1024/1.rimg')
    parser.add_argument('-cp', '--checkpoint',
                        type=str,
                        required=True,
                        help='Check point filename. [.pth]')
    parser.add_argument('-v', '--voxel_size',
                        type=float,
                        required=False,
                        default=0.0,
                        help='Voxel size for visualization, Default: no voxelization')
    args = parser.parse_args()

    # ROS settings for visualization
    rospy.init_node('demo_resolution_free_lidar')
    points_publisher = rospy.Publisher('/demo_resolution_free_lidar/points', PointCloud2, queue_size=1)

    # Load the check point
    check_point = torch.load(args.checkpoint)

    # Model
    model = generate_model(check_point['model']['name'], check_point['model']['args'])
    model.load_state_dict(check_point['model']['state_dict'])
    if torch.cuda.device_count() > 1:
        model = nn.parallel.DataParallel(model)
    model.eval().cuda()

    # Prepare the trained LiDAR specification
    lidar_in = check_point['lidar_in']
    lidar_out = copy.deepcopy(check_point['lidar_in'])

    # Set the various resolutions
    # DEMO: [16 x 1024] --> [64 x 1024] --> [2048 x 4096]
    v_channels = np.concatenate((np.arange(16, 64, 2, dtype=int), (np.arange(1.0, 4.1, 0.1, dtype=float) * 64.0).astype(int)))
    h_channels = np.concatenate((np.repeat([1024], 24), (np.arange(1.0, 4.1, 0.1, dtype=float) * 1024.0).astype(int)))

    print("===================== Demo Configuration ======================  ")
    check_point_filename = args.checkpoint
    model_name = check_point['model']['name']
    input_filename = args.input_filename
    print('  Check point file:', check_point_filename)
    print('  Model:', model_name, '(' + str(sum(p.numel() for p in model.parameters() if p.requires_grad)) + ' parameters)')
    for key, value in check_point['model']['args'].items():
        print('    ' + key + ':', value)
    print('  ')
    print('  Input filename:', args.input_filename)
    print('  Input resolution:', lidar_in['channels'], 'x', lidar_in['points_per_ring'])
    print('  Voxel size:', args.voxel_size)
    print("===============================================================  \n")

    # Read the input range image
    input_range_image = read_range_image_binary(input_filename, lidar=lidar_in)
    input_range_image = normalization_ranges(input_range_image, norm_r=lidar_in['norm_r'])

    res_idx = 0
    while not rospy.is_shutdown():
        # Select the vertical and horizontal resolutions
        lidar_out['channels'] = v_channels[res_idx]
        lidar_out['points_per_ring'] = h_channels[res_idx]

        # Generate the normalized queries according to width and height
        query_lasers = generate_laser_directions(lidar_out)
        query_lasers = normalization_queries(query_lasers, lidar_in)

        # Reconstruct the high-resolution range image
        pred_detection_distances = predict_detection_distances(input_image=input_range_image,
                                                               pixel_centers=query_lasers,
                                                               pred_batch=100000)
        pred_range_image = pred_detection_distances.reshape(lidar_out['channels'], lidar_out['points_per_ring'])
        pred_range_image = denormalization_ranges(range_image=pred_range_image, norm_r=lidar_out['norm_r'])

        # Convert the range image to points
        pred_range_image[pred_range_image < lidar_out['min_r']] = 0.0
        pred_range_image[pred_range_image > lidar_out['max_r']] = 0.0
        points = range_image_to_points(range_image=pred_range_image, lidar=lidar_out, remove_zero_range=True)

        # Option: voxelization
        if args.voxel_size > 0.0:
            points = voxelizer.get_voxel_centers_from_points(points, args.voxel_size)

        # Visualize the points in RViz
        pointcloud_msg = get_pointcloud_xyz(points)
        if points_publisher.get_num_connections() > 0:
            points_publisher.publish(pointcloud_msg)

        print('Target resolution:', lidar_out['channels'], 'x', lidar_out['points_per_ring'], '(' + str(points.shape[0]) + ' points)')

        res_idx = (res_idx + 1) % len(v_channels)

        time.sleep(0.1)

import os
import numpy as np
import argparse
import math
import cv2
import torch
# from pyemd import emd

offset_lut = np.array([48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0])

azimuth_lut = np.array([4.23,1.43,-1.38,-4.18,4.23,1.43,-1.38,-4.18,4.24,1.43,-1.38,-4.18,4.24,1.42,-1.38,-4.19,4.23,1.43,-1.38,-4.19,4.23,1.43,-1.39,-4.19,4.23,1.42,-1.39,-4.2,4.23,1.43,-1.39,-4.19,4.23,1.42,-1.4,-4.2,4.23,1.42,-1.4,-4.2,4.22,1.41,-1.4,-4.21,4.22,1.41,-1.39,-4.2,4.22,1.41,-1.4,-4.21,4.22,1.41,-1.4,-4.21,4.22,1.41,-1.4,-4.21,4.22,1.41,-1.41,-4.21,4.22,1.41,-1.41,-4.21,4.21,1.4,-1.41,-4.21,4.21,1.41,-1.41,-4.21,4.22,1.41,-1.42,-4.22,4.22,1.4,-1.41,-4.22,4.21,1.41,-1.42,-4.22,4.22,1.4,-1.41,-4.22,4.21,1.4,-1.41,-4.23,4.21,1.4,-1.42,-4.23,4.21,1.4,-1.42,-4.22,4.21,1.39,-1.42,-4.22,4.21,1.4,-1.42,-4.21,4.21,1.4,-1.42,-4.22,4.2,1.4,-1.41,-4.22,4.2,1.4,-1.42,-4.22,4.2,1.4,-1.42,-4.22])

elevation_lut = np.array([21.42,21.12,20.81,20.5,20.2,19.9,19.58,19.26,18.95,18.65,18.33,18.02,17.68,17.37,17.05,16.73,16.4,16.08,15.76,15.43,15.1,14.77,14.45,14.11,13.78,13.45,13.13,12.79,12.44,12.12,11.77,11.45,11.1,10.77,10.43,10.1,9.74,9.4,9.06,8.72,8.36,8.02,7.68,7.34,6.98,6.63,6.29,5.95,5.6,5.25,4.9,4.55,4.19,3.85,3.49,3.15,2.79,2.44,2.1,1.75,1.38,1.03,0.68,0.33,-0.03,-0.38,-0.73,-1.07,-1.45,-1.8,-2.14,-2.49,-2.85,-3.19,-3.54,-3.88,-4.26,-4.6,-4.95,-5.29,-5.66,-6.01,-6.34,-6.69,-7.05,-7.39,-7.73,-8.08,-8.44,-8.78,-9.12,-9.45,-9.82,-10.16,-10.5,-10.82,-11.19,-11.52,-11.85,-12.18,-12.54,-12.87,-13.2,-13.52,-13.88,-14.21,-14.53,-14.85,-15.2,-15.53,-15.84,-16.16,-16.5,-16.83,-17.14,-17.45,-17.8,-18.11,-18.42,-18.72,-19.06,-19.37,-19.68,-19.97,-20.31,-20.61,-20.92,-21.22])

origin_offset = 0.015806

lidar_to_sensor_z_offset = 0.03618

angle_off = math.pi * 4.2285/180.

def idx_from_px(px, cols):
    vv = (int(px[0]) + cols - offset_lut[int(px[1])]) % cols
    idx = px[1] * cols + vv
    return idx

def px_to_xyz(px, p_range, cols):
    u = (cols + px[0]) % cols
    azimuth_radians = math.pi * 2.0 / cols 
    encoder = 2.0 * math.pi - (u * azimuth_radians) 
    azimuth = angle_off
    elevation = math.pi * elevation_lut[int(px[1])] / 180.
    x_lidar = (p_range - origin_offset) * math.cos(encoder+azimuth)*math.cos(elevation) + origin_offset*math.cos(encoder)
    y_lidar = (p_range - origin_offset) * math.sin(encoder+azimuth)*math.cos(elevation) + origin_offset*math.sin(encoder)
    z_lidar = (p_range - origin_offset) * math.sin(elevation) 
    x_sensor = -x_lidar
    y_sensor = -y_lidar
    z_sensor = z_lidar + lidar_to_sensor_z_offset
    return np.array([x_sensor, y_sensor, z_sensor])

def img_to_pcd(img_range, maximum_range = 200):
    rows, cols = img_range.shape[:2]

    points = np.zeros((rows*cols, 3))
    for u in range(cols):
        for v in range(rows):

            idx = idx_from_px((u, v), cols)
            range_px = img_range[v, u] * maximum_range
            if range_px < 0.1:
                continue
            else:
                point_repro = px_to_xyz((u,v), range_px, cols)
                points[idx, :] = point_repro


    return points



# Durlar Dataset pixel in range map -> index in 
# def idx_from_px(px, cols):
#     vv = (int(px[0]) + cols - offset_lut[int(px[1])]) % cols
#     idx = px[1] * cols + vv
#     return idx

# def px_from_idx(idx, cols):
#     vv = idx % cols
#     y = math.ceil((idx-vv) / cols)
#     x = vv + offset_lut[y]
#     if x >= cols:
#         x = x - cols
#     return (x, y)


def mean_absolute_error(pred_img, gt_img):
    abs_error = (pred_img - gt_img).abs()

    return abs_error.mean()

from scipy.spatial import distance
from scipy.spatial import cKDTree

def chamfer_distance(points1, points2):
    tree = cKDTree(points2)
    _, min_dist_1_index = tree.query(points1)
    min_dist_1 = np.linalg.norm(points1 - points2[min_dist_1_index], axis = 1)
    tree = cKDTree(points1)
    _, min_dist_2_index = tree.query(points2)
    min_dist_2 = np.linalg.norm(points2 - points1[min_dist_2_index], axis = 1)
    chamfer_dist = np.mean(min_dist_1) + np.mean(min_dist_2)
    return chamfer_dist

def chamfer_distance_2(points1, points2):
    # Calculate the distance from each point in points1 to its nearest neighbor in points2
    dist1 = distance.cdist(points1, points2).min(axis=1)
    # Calculate the distance from each point in points2 to its nearest neighbor in points1
    dist2 = distance.cdist(points2, points1).min(axis=1)
    # Calculate the average distance
    chamfer_dist = np.mean(dist1) + np.mean(dist2)
    return chamfer_dist

# def earth_mover_distance(point_cloud1, point_cloud2):
#     distance_matrix = distance.cdist(point_cloud1, point_cloud2)
#     print(distance_matrix.shape)
#     num_bins = np.ones(len(point_cloud1), dtype = np.float64) / len(point_cloud1)
#     emd_ = emd(num_bins, num_bins, distance_matrix)
#     return emd_

# def 

def voxelize_point_cloud(point_cloud, grid_size, min_coord, max_coord):
    # Calculate the dimensions of the voxel grid
    dimensions = ((max_coord - min_coord) / grid_size).astype(int) + 1

    # Create the voxel grid
    voxel_grid = np.zeros(dimensions, dtype=bool)

    # Assign points to voxels
    indices = ((point_cloud - min_coord) / grid_size).astype(int)
    voxel_grid[tuple(indices.T)] = True

    return voxel_grid

def calculate_metrics(voxel_grid_predicted, voxel_grid_ground_truth):
    intersection = np.logical_and(voxel_grid_predicted, voxel_grid_ground_truth)
    union = np.logical_or(voxel_grid_predicted, voxel_grid_ground_truth)

    iou = np.sum(intersection) / np.sum(union)

    true_positive = np.sum(intersection)
    false_positive = np.sum(voxel_grid_predicted) - true_positive
    false_negative = np.sum(voxel_grid_ground_truth) - true_positive

    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)

    return iou, precision, recall

def inverse_huber_loss(output, target):
    absdiff = torch.abs(output-target)
    C = 0.2*torch.max(absdiff).item()
    return torch.where(absdiff < C, absdiff,(absdiff*absdiff+C*C)/(2*C))

if __name__ == "__main__":
    point1 = np.random.uniform(0, 1, (100, 3))
    point2 = np.random.uniform(0, 1, (100, 3))

    point3 = point1.copy()

    point3[0, 0] += 0.1

    cd_1 = chamfer_distance(point1, point2)
    cd_2 = chamfer_distance_2(point1, point2)



    print(cd_1, cd_2)


    # point4 = np.array([[0.5, 1, 1],
    #                    [1, 1, 1],
    #                    [1, 1, 1.5]])
    
    # point5 = np.array([[1, 1, 1],
    #                    [1, 1, 1],
    #                    [1, 1, 1]])
    

    min_coord = np.min(np.vstack((point1, point3)), axis=0)
    max_coord = np.max(np.vstack((point1, point3)), axis=0)
    grid_size = 0.1
    # Voxelize the ground truth and prediction point clouds
    voxel_grid_predicted = voxelize_point_cloud(point1, grid_size, min_coord, max_coord)
    voxel_grid_ground_truth = voxelize_point_cloud(point3, grid_size, min_coord, max_coord)

    print(voxel_grid_ground_truth.shape, voxel_grid_predicted.shape)

    # Calculate metrics
    iou, precision, recall = calculate_metrics(voxel_grid_predicted, voxel_grid_ground_truth)

    

    # grids_pred = create_grid(point4, grid_length=1)
    # grids_gt = create_grid(point5, grid_length=1)
    # grids = create_grid_around_point(point4, spacing=0.1)


    print(iou, precision, recall)
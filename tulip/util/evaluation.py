import numpy as np
import math
import torch
from chamfer_distance import ChamferDistance as chamfer_dist
# from pyemd import emd

offset_lut = np.array([48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0,48,32,16,0])

azimuth_lut = np.array([4.23,1.43,-1.38,-4.18,4.23,1.43,-1.38,-4.18,4.24,1.43,-1.38,-4.18,4.24,1.42,-1.38,-4.19,4.23,1.43,-1.38,-4.19,4.23,1.43,-1.39,-4.19,4.23,1.42,-1.39,-4.2,4.23,1.43,-1.39,-4.19,4.23,1.42,-1.4,-4.2,4.23,1.42,-1.4,-4.2,4.22,1.41,-1.4,-4.21,4.22,1.41,-1.39,-4.2,4.22,1.41,-1.4,-4.21,4.22,1.41,-1.4,-4.21,4.22,1.41,-1.4,-4.21,4.22,1.41,-1.41,-4.21,4.22,1.41,-1.41,-4.21,4.21,1.4,-1.41,-4.21,4.21,1.41,-1.41,-4.21,4.22,1.41,-1.42,-4.22,4.22,1.4,-1.41,-4.22,4.21,1.41,-1.42,-4.22,4.22,1.4,-1.41,-4.22,4.21,1.4,-1.41,-4.23,4.21,1.4,-1.42,-4.23,4.21,1.4,-1.42,-4.22,4.21,1.39,-1.42,-4.22,4.21,1.4,-1.42,-4.21,4.21,1.4,-1.42,-4.22,4.2,1.4,-1.41,-4.22,4.2,1.4,-1.42,-4.22,4.2,1.4,-1.42,-4.22])

elevation_lut = np.array([21.42,21.12,20.81,20.5,20.2,19.9,19.58,19.26,18.95,18.65,18.33,18.02,17.68,17.37,17.05,16.73,16.4,16.08,15.76,15.43,15.1,14.77,14.45,14.11,13.78,13.45,13.13,12.79,12.44,12.12,11.77,11.45,11.1,10.77,10.43,10.1,9.74,9.4,9.06,8.72,8.36,8.02,7.68,7.34,6.98,6.63,6.29,5.95,5.6,5.25,4.9,4.55,4.19,3.85,3.49,3.15,2.79,2.44,2.1,1.75,1.38,1.03,0.68,0.33,-0.03,-0.38,-0.73,-1.07,-1.45,-1.8,-2.14,-2.49,-2.85,-3.19,-3.54,-3.88,-4.26,-4.6,-4.95,-5.29,-5.66,-6.01,-6.34,-6.69,-7.05,-7.39,-7.73,-8.08,-8.44,-8.78,-9.12,-9.45,-9.82,-10.16,-10.5,-10.82,-11.19,-11.52,-11.85,-12.18,-12.54,-12.87,-13.2,-13.52,-13.88,-14.21,-14.53,-14.85,-15.2,-15.53,-15.84,-16.16,-16.5,-16.83,-17.14,-17.45,-17.8,-18.11,-18.42,-18.72,-19.06,-19.37,-19.68,-19.97,-20.31,-20.61,-20.92,-21.22])

origin_offset = 0.015806

lidar_to_sensor_z_offset = 0.03618

angle_off = math.pi * 4.2285/180.

def idx_from_px(px, cols):
    vv = (px[:,0].astype(int) + cols - offset_lut[px[:, 1].astype(int)]) % cols
    idx = px[:, 1] * cols + vv
    return idx


def px_to_xyz(px, p_range, cols): # px: (u, v) size = (H*W,2)
    u = (cols + px[:,0]) % cols
    azimuth_radians = math.pi * 2.0 / cols 
    encoder = 2.0 * math.pi - (u * azimuth_radians) 
    azimuth = angle_off
    elevation = math.pi * elevation_lut[px[:, 1].astype(int)] / 180.

    x_lidar = (p_range - origin_offset) * np.cos(encoder+azimuth)*np.cos(elevation) + origin_offset*np.cos(encoder)
    y_lidar = (p_range - origin_offset) * np.sin(encoder+azimuth)*np.cos(elevation) + origin_offset*np.sin(encoder)
    z_lidar = (p_range - origin_offset) * np.sin(elevation) 
    x_sensor = -x_lidar
    y_sensor = -y_lidar
    z_sensor = z_lidar + lidar_to_sensor_z_offset
    return np.stack((x_sensor, y_sensor, z_sensor), axis=-1)

def img_to_pcd_durlar(img_range, maximum_range = 120):  # 1 x H x W cuda torch
    rows, cols = img_range.shape[:2]
    uu, vv = np.meshgrid(np.arange(cols), np.arange(rows), indexing="ij")
    uvs = np.stack((uu, vv), axis=-1).reshape(-1, 2)

    points = np.zeros((rows*cols, 3))
    indices = idx_from_px(uvs, cols)
    points_all = px_to_xyz(uvs, img_range.transpose().reshape(-1) * maximum_range, cols)

    points[indices, :] = points_all
    return points

def img_to_pcd_kitti(img_range, maximum_range = 120, low_res = False, intensity = None):
    if low_res:
        image_rows = 16
    else:
        image_rows = 64
    image_cols = 1024
    ang_start_y = 24.8
    ang_res_y = 26.8 / (image_rows -1)
    ang_res_x = 360 / image_cols

    rowList = []
    colList = []
    for i in range(image_rows):
        rowList = np.append(rowList, np.ones(image_cols)*i)
        colList = np.append(colList, np.arange(image_cols))


    verticalAngle = np.float32(rowList * ang_res_y) - ang_start_y
    horizonAngle = - np.float32(colList + 1 - (image_cols/2)) * ang_res_x + 90.0
    
    verticalAngle = verticalAngle / 180.0 * np.pi
    horizonAngle = horizonAngle / 180.0 * np.pi


    lengthList = img_range.reshape(image_rows*image_cols) * maximum_range

    x = np.sin(horizonAngle) * np.cos(verticalAngle) * lengthList
    y = np.cos(horizonAngle) * np.cos(verticalAngle) * lengthList
    z = np.sin(verticalAngle) * lengthList
    if intensity is not None:
        intensity = intensity.reshape(image_rows*image_cols)
        points = np.column_stack((x,y,z,intensity))
    else:    
        points = np.column_stack((x,y,z))

    return points


def img_to_pcd_carla(img_range, maximum_range = 80):
    # img_range = np.flip(img_range)
    rows, cols = img_range.shape[:2]

    v_dir = np.linspace(start=-15, stop=15, num=rows)
    h_dir = np.linspace(start=-180, stop=180, num=cols, endpoint=False)

    v_angles = []
    h_angles = []

    for i in range(rows):
        v_angles = np.append(v_angles, np.ones(cols) * v_dir[i])
        h_angles = np.append(h_angles, h_dir)

    angles = np.stack((v_angles, h_angles), axis=-1).astype(np.float32)
    angles = np.deg2rad(angles)

    r = img_range.flatten() * maximum_range


    x = np.sin(angles[:, 1]) * np.cos(angles[:, 0]) * r
    y = np.cos(angles[:, 1]) * np.cos(angles[:, 0]) * r
    z = np.sin(angles[:, 0]) * r

    points = np.stack((x, y, z), axis=-1)

    return points


def mean_absolute_error(pred_img, gt_img):
    abs_error = (pred_img - gt_img).abs()

    return abs_error.mean()


def chamfer_distance(points1, points2, num_points = None):
    source = torch.from_numpy(points1[None, :]).cuda()
    target = torch.from_numpy(points2[None, :]).cuda()


    chd = chamfer_dist()
    dist1, dist2, _, _ = chd(source, target)
    cdist = (torch.mean(dist1)) + (torch.mean(dist2)) if num_points is None else (dist1.sum()/num_points) + (dist2.sum()/num_points)

    return cdist.detach().cpu()

def depth_wise_unconcate(imgs): # H W
    b, c, h, w = imgs.shape
    new_imgs = torch.zeros((b, h*c, w)).cuda()
    low_res_indices = [range(i, h*c+i, c) for i in range(c)]


    for i, indices in enumerate(low_res_indices):
        new_imgs[:, indices,:] = imgs[:, i, :, :]

    return new_imgs.reshape(b, 1, h*c, w)


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

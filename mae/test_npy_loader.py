
from util.datasets import build_carla_upsampling_dataset, RandomRollRangeMap, build_durlar_upsampling_dataset
import os
import numpy as np
import argparse
import cv2
import matplotlib.pyplot as plt
import trimesh

# np.random.seed(0)

# a = RandomRollRangeMap()
# b = a.shift


# args = argparse.ArgumentParser('MAE pre-training', add_help=False)

# args.data_path = "/cluster/work/riner/users/biyang/dataset/depth_intensity_large"
# args.crop = False
# args.in_chans = 1
# args.log_transform = False
# args.roll = False

# args.img_size_low_res = (32, 2048)
# args.img_size_high_res = (128, 2048)


# dataset = build_carla_upsampling_dataset(is_train = False, args = args)

# input = dataset[0][0][0]
# output = dataset[0][1][0]

def normalize_depth(val, min_v, max_v):
    """ 
    print 'nomalized depth value' 
    nomalize values to 0-255 & close distance value has high value. (similar to stereo vision's disparity map)
    """
    return (((max_v - val) / (max_v - min_v)) * 255).astype(np.uint8)

def normalize_val(val, min_v, max_v):
    """ 
    print 'nomalized depth value' 
    nomalize values to 0-255 & close distance value has low value.
    """
    return (((val - min_v) / (max_v - min_v)) * 255).astype(np.uint8)

def in_h_range_points(m, n, fov):
    """ extract horizontal in-range points """
    return np.logical_and(np.arctan2(n,m) > (-fov[1] * np.pi / 180), \
                          np.arctan2(n,m) < (-fov[0] * np.pi / 180))

def in_v_range_points(m, n, fov):
    """ extract vertical in-range points """
    return np.logical_and(np.arctan2(n,m) < (fov[1] * np.pi / 180), \
                          np.arctan2(n,m) > (fov[0] * np.pi / 180))

def fov_setting(points, x, y, z, dist, h_fov, v_fov):
    """ filter points based on h,v FOV  """
    
    if h_fov[1] == 180 and h_fov[0] == -180 and v_fov[1] == 2.0 and v_fov[0] == -24.9:
        return points
    
    if h_fov[1] == 180 and h_fov[0] == -180:
        return points[in_v_range_points(dist, z, v_fov)]
    elif v_fov[1] == 2.0 and v_fov[0] == -24.9:        
        return points[in_h_range_points(x, y, h_fov)]
    else:
        h_points = in_h_range_points(x, y, h_fov)
        v_points = in_v_range_points(dist, z, v_fov)
        return points[np.logical_and(h_points, v_points)]

def reproject_range_map(range_map, image_rows_high, image_cols, ang_start_y, ang_res_y, ang_res_x):

    rowList = []
    colList = []
    for i in range(image_rows_high):
        rowList = np.append(rowList, np.ones(image_cols)*i)
        colList = np.append(colList, np.arange(image_cols))


    # uvs = np.stack((uu, vv), axis=-1).reshape(-1, 2)


    verticalAngle = np.float32(rowList * ang_res_y) - ang_start_y
    horizonAngle = - np.float32(colList + 1 - (image_cols/2)) * ang_res_x + 90.0
    
    verticalAngle = verticalAngle / 180.0 * np.pi
    horizonAngle = horizonAngle / 180.0 * np.pi


    lengthList = range_map.reshape(image_rows_high*image_cols)

    x = np.sin(horizonAngle) * np.cos(verticalAngle) * lengthList
    y = np.cos(horizonAngle) * np.cos(verticalAngle) * lengthList
    z = np.sin(verticalAngle) * lengthList


    
        
    points = np.column_stack((x,y,z))

    return points
    
def create_range_map(points_array, image_rows_full, image_cols, ang_start_y, ang_res_y, ang_res_x, max_range, min_range):
    #print('processing {}th point cloud message...\r'.format(range_image_array.shape[0])),
    range_image = np.zeros((image_rows_full, image_cols, 1), dtype=np.float32)
    intensity_map = np.zeros((image_rows_full, image_cols, 1), dtype=np.float32)
    x = points_array[:,0]
    y = points_array[:,1]
    z = points_array[:,2]
    intensity = points_array[:, 3]
    # find row id

    vertical_angle = np.arctan2(z, np.sqrt(x * x + y * y)) * 180.0 / np.pi
    relative_vertical_angle = vertical_angle + ang_start_y
    rowId = np.int_(np.round_(relative_vertical_angle / ang_res_y))
    # find column id
    # Inverse sign of y for kitti data
    horitontal_angle = np.arctan2(x, y) * 180.0 / np.pi
    # horitontal_angle = np.arctan2(-y, x) * 180.0 / np.pi

    colId = -np.int_((horitontal_angle-90.0)/ang_res_x) + image_cols/2
    # colId = -np.int_(horitontal_angle / ang_res_x) + image_cols / 2

    shift_ids = np.where(colId>=image_cols)
    colId[shift_ids] = colId[shift_ids] - image_cols
    colId = colId.astype(np.int64)
    # filter range
    thisRange = np.sqrt(x * x + y * y + z * z)
    thisRange[thisRange > max_range] = 0
    thisRange[thisRange < min_range] = 0

    # filter Internsity
    intensity[thisRange > max_range] = 0
    intensity[thisRange < min_range] = 0


    valid_scan = (rowId >= 0) & (rowId < image_rows_full) & (colId >= 0) & (colId < image_cols)

    rowId_valid = rowId[valid_scan]
    colId_valid = colId[valid_scan]
    thisRange_valid = thisRange[valid_scan]
    intensity_valid = intensity[valid_scan]



    range_image[rowId_valid, colId_valid, :] = thisRange_valid.reshape(-1, 1)
    intensity_map[rowId_valid, colId_valid, :] = intensity_valid.reshape(-1, 1)

    # # save range info to range image
    # for i in range(len(thisRange)):
    #     if rowId[i] < 0 or rowId[i] >= image_rows_full or colId[i] < 0 or colId[i] >= image_cols:
    #         continue
    #     range_image[rowId[i], colId[i], :] = thisRange[i]
    #     intensity_map[rowId[i], colId[i], :] = intensity[i]

    lidar_data_projected = np.concatenate((range_image, intensity_map), axis = -1)

    return lidar_data_projected

def load_from_bin(bin_path):
    lidar_data = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    # ignore reflectivity info
    return lidar_data


velodyne_data = "/cluster/work/riner/users/biyang/dataset/KITTI/2011_09_26/2011_09_26_drive_0005_sync/velodyne_points/data/0000000000.bin"
velo_points = load_from_bin(velodyne_data)

# velo_points = velo_points[:, :3]

# print(velo_points.shape)
# exit(0)

pcd = trimesh.PointCloud(vertices=velo_points[:, :3])
pcd.export("kitti.ply")

v_fov, h_fov = (-24.8, 2.0), (-180,180)

ang_res_y = 26.8 / (64-1)
ang_res_x = 360 / 1024
max_range = 120
min_range = 0
# pano_img = velo_points_2_pano(velo_points, v_res=0.42, h_res=0.35, v_fov=v_fov, h_fov=h_fov, depth = False)
pano_img = create_range_map(velo_points, image_rows_full = 64, image_cols = 1024, ang_start_y = 24.8, ang_res_y = ang_res_y, ang_res_x = ang_res_x, max_range = max_range, min_range = min_range)

pano_img = pano_img[:, :, 0]
pano_img = pano_img.reshape(64, 1024)





# display result image
# plt.subplots(1,1, figsize = (13,3) )
# plt.title("Result of Vertical FOV ({} , {}) & Horizontal FOV ({} , {})".format(v_fov[0],v_fov[1],h_fov[0],h_fov[1]))
plt.imshow(pano_img)
# plt.axis('off')
plt.savefig('kitti.png')

print(pano_img.shape)


reproject_points = reproject_range_map(range_map=pano_img, image_rows_high=64, image_cols=1024, ang_start_y=24.8, ang_res_y=ang_res_y, ang_res_x=ang_res_x)

reproject_pcd = trimesh.PointCloud(vertices = reproject_points)
reproject_pcd.export("kitti_reprojected.ply")







exit(0)



args = argparse.ArgumentParser('MAE pre-training', add_help=False)

args.data_path = "/cluster/work/riner/users/biyang/dataset/Carla"
args.crop = False
args.in_chans = 1
args.log_transform = False

args.img_size_low_res = (16, 1024)
args.img_size_high_res = (128, 2048)

# dataset_train = build_durlar_dataset(is_train = True, args = args)
# dataset_val = build_durlar_dataset(is_train = False, args = args)

dataset = build_carla_upsampling_dataset(is_train = True, args = args)

input = dataset[0][0][0]
output = dataset[0][1][0]

img = output.numpy().transpose(1, 2, 0)
img_2 = input.numpy().transpose(1, 2, 0)

cv2.imwrite('carla.png', img*255)
cv2.imwrite('carla_low.png', img_2*255)


# for data in dataset_train:
#     a = data[0]

#     print(a.max(), a.min())

#     print((a>1).sum())


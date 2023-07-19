import torch
import trimesh
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import os
import argparse
from bin_to_img import *
import cv2


# global image_rows_full, image_cols, ang_start_y, ang_res_y,  ang_res_x, max_range, min_range
# # range image size, depends on your sensor, i.e., VLP-16: 16x1800, OS1-64: 64x1024
# image_rows_full = 16
# image_cols = 1024

# # Ouster OS1-64 (gen1)
# ang_res_x = 360.0/float(image_cols) # horizontal resolution
# ang_res_y = 35/float(image_rows_full-1) # vertical resolution
# ang_start_y = 25 # bottom beam angle
# max_range = 100.0
# min_range = 2.0

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=["val", "train", "test"])
    parser.add_argument("--input_path", type=str , default=None)


    parser.add_argument("--rows", type = int, default = 128)
    parser.add_argument("--cols", type = int, default = 2048)
    parser.add_argument("--color", action="store_true", help="Use colormap to map the depth value to rgb channel")
    parser.add_argument("--normalize", action="store_true", help="Normalize range map with LiDAR max range")

   
    return parser.parse_args()

# def create_range_map(points_array):
#     #print('processing {}th point cloud message...\r'.format(range_image_array.shape[0])),
#     range_image = np.zeros((1, image_rows_full, image_cols, 1), dtype=np.float32)
#     intensity_map = np.zeros((1, image_rows_full, image_cols, 1), dtype=np.float32)
#     x = points_array[:,0]
#     y = points_array[:,1]
#     z = points_array[:,2]
#     intensity = points_array[:, 3]
#     # find row id
#     vertical_angle = np.arctan2(z, np.sqrt(x * x + y * y)) * 180.0 / np.pi
#     relative_vertical_angle = vertical_angle + ang_start_y
#     rowId = np.int_(np.round_(relative_vertical_angle / ang_res_y))
#     # find column id
#     horitontal_angle = np.arctan2(x, y) * 180.0 / np.pi
#     colId = -np.int_((horitontal_angle-90.0)/ang_res_x) + image_cols/2
#     shift_ids = np.where(colId>=image_cols)
#     colId[shift_ids] = colId[shift_ids] - image_cols
#     colId = colId.astype(np.int64)
#     # filter range
#     thisRange = np.sqrt(x * x + y * y + z * z)
#     thisRange[thisRange > max_range] = 0
#     thisRange[thisRange < min_range] = 0

#     # filter Internsity
#     intensity[thisRange > max_range] = 0
#     intensity[thisRange < min_range] = 0

#     # save range info to range image
#     for i in range(len(thisRange)):
#         if rowId[i] < 0 or rowId[i] >= image_rows_full or colId[i] < 0 or colId[i] >= image_cols:
#             continue
#         range_image[0, rowId[i], colId[i], 0] = thisRange[i]
#         intensity_map[0, rowId[i], colId[i], 0] = intensity[i]
#     # append range image to array
#     #range_image_array = np.append(range_image_array, range_image, axis=0)

#     return range_image, intensity_map

# def recover_pcd(thisImage, height = 0, color = [255, 0 ,0]):
#         # multi-channel range image, the first channel is range
#     if len(thisImage.shape) == 3:
#         thisImage = thisImage[:,:,0]

#     rowList = []
#     colList = []
#     for i in range(image_rows_full):
#         rowList = np.append(rowList, np.ones(image_cols)*i)
#         colList = np.append(colList, np.arange(image_cols))

#     verticalAngle = np.float32(rowList * ang_res_y) - ang_start_y
#     horizonAngle = - np.float32(colList + 1 - (image_cols/2)) * ang_res_x + 90.0
    
#     verticalAngle = verticalAngle / 180.0 * np.pi
#     horizonAngle = horizonAngle / 180.0 * np.pi


#     lengthList = thisImage.reshape(image_rows_full*image_cols)
#     lengthList[lengthList > max_range] = 0.0
#     lengthList[lengthList < min_range] = 0.0

#     x = np.sin(horizonAngle) * np.cos(verticalAngle) * lengthList
#     y = np.cos(horizonAngle) * np.cos(verticalAngle) * lengthList
#     z = np.sin(verticalAngle) * lengthList + height
    
#     points = np.column_stack((x,y,z))
#     # delete points that has range value 0
#     points = np.delete(points, np.where(lengthList==0), axis=0) # comment this line for visualize at the same speed (for video generation)

#     colors = np.tile(color, (len(points), 1))
#     pcd = trimesh.PointCloud(vertices = points, colors = colors)
#     return pcd

# Ouster Maximum Range
max_range_durlar = 200

flags = read_args()
if flags.color:
    cNorm = colors.Normalize(vmin=0, vmax=max_range_durlar)
    jet = plt.get_cmap('magma')
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
path_dataset = flags.input_path
ouster_points_path = os.path.join(path_dataset,"ouster_points/data")


intensity_output = os.path.join(path_dataset,"intensity")
depth_output = os.path.join(path_dataset,"depth")
output_folders = [intensity_output, depth_output] 
if flags.color:
    depth_colored_output = os.path.join(path_dataset, "depth_colored")
    output_folders.append(depth_colored_output)
for path in output_folders:
    if not os.path.exists(path):
        print("There is no output folder, creating one")
        os.mkdir(path)

files = os.listdir(ouster_points_path)
files.sort()
for i, bin_filepath in enumerate(files):
    if i <= 40000:
        continue
    # if i > 42000:
    #     break
    fullpath = os.path.join(ouster_points_path, bin_filepath)
    scan = (np.fromfile(fullpath, dtype=np.float32)).reshape(-1, 4)
    # range_map, intensity, avg_repro_error, max_repro_error, px_max_repro_error = main(scan=scan, rows=flags.rows, cols = flags.cols)
    range_map, intensity = main(scan=scan, rows=flags.rows, cols = flags.cols)


    # if avg_repro_error > 0.005 or max_repro_error > 0.05:
    #     print("Filename: ", bin_filepath)
    #     print("Average Reprojection Error: ", avg_repro_error, ", Maximum Reprojection Error: ", max_repro_error)
    #     print("The pixel of the maximum reprojection error: ", px_max_repro_error)


    if flags.normalize:
        range_map = (range_map/200) * 255

    output_name = bin_filepath.replace(".bin", ".png")
    if flags.color:
        # cNorm = colors.Normalize(vmin=np.min(range_map), vmax=np.max(range_map))
        # jet = plt.get_cmap('jet')
        # scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

        range_map_colored = scalarMap.to_rgba(range_map)[...,:3]*255
        cv2.imwrite(os.path.join(depth_colored_output, output_name), range_map_colored)
    
    cv2.imwrite(os.path.join(intensity_output, output_name), intensity)
    cv2.imwrite(os.path.join(depth_output, output_name), range_map)

print("Done")

# test = os.path.join(ouster_pcd_path, os.listdir(ouster_pcd_path)[0])
# scan = (np.fromfile(test, dtype=np.float32)).reshape(-1, 4)

# intensity, rangemap = bin_to_img(scan=scan, rows=flags.rows, cols = flags.cols)

from codd import CODDAggSnippet, CODDAggDataset
import torch
import trimesh
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import os
import argparse


global image_rows_full, image_cols, ang_start_y, ang_res_y,  ang_res_x, max_range, min_range
# range image size, depends on your sensor, i.e., VLP-16: 16x1800, OS1-64: 64x1024
image_rows_full = 16
image_cols = 1024

# Ouster OS1-64 (gen1)
ang_res_x = 360.0/float(image_cols) # horizontal resolution
ang_res_y = 35/float(image_rows_full-1) # vertical resolution
ang_start_y = 25 # bottom beam angle
max_range = 100.0
min_range = 2.0

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=["val", "train", "test"])
    parser.add_argument('--output_folder', type=str, default="output")
    parser.add_argument("--save_data", action="store_true", default=False)
    parser.add_argument("--input_path", type=str , default="/cluster/scratch/biyang/data")

   
    return parser.parse_args()

def create_range_map(points_array):
    #print('processing {}th point cloud message...\r'.format(range_image_array.shape[0])),
    range_image = np.zeros((1, image_rows_full, image_cols, 1), dtype=np.float32)
    intensity_map = np.zeros((1, image_rows_full, image_cols, 1), dtype=np.float32)
    x = points_array[:,0]
    y = points_array[:,1]
    z = points_array[:,2]
    intensity = points_array[:, 3]
    # find row id
    vertical_angle = np.arctan2(z, np.sqrt(x * x + y * y)) * 180.0 / np.pi
    relative_vertical_angle = vertical_angle + ang_start_y
    rowId = np.int_(np.round_(relative_vertical_angle / ang_res_y))
    # find column id
    horitontal_angle = np.arctan2(x, y) * 180.0 / np.pi
    colId = -np.int_((horitontal_angle-90.0)/ang_res_x) + image_cols/2
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

    # save range info to range image
    for i in range(len(thisRange)):
        if rowId[i] < 0 or rowId[i] >= image_rows_full or colId[i] < 0 or colId[i] >= image_cols:
            continue
        range_image[0, rowId[i], colId[i], 0] = thisRange[i]
        intensity_map[0, rowId[i], colId[i], 0] = intensity[i]
    # append range image to array
    #range_image_array = np.append(range_image_array, range_image, axis=0)

    return range_image, intensity_map

def recover_pcd(thisImage, height = 0, color = [255, 0 ,0]):
        # multi-channel range image, the first channel is range
    if len(thisImage.shape) == 3:
        thisImage = thisImage[:,:,0]

    rowList = []
    colList = []
    for i in range(image_rows_full):
        rowList = np.append(rowList, np.ones(image_cols)*i)
        colList = np.append(colList, np.arange(image_cols))

    verticalAngle = np.float32(rowList * ang_res_y) - ang_start_y
    horizonAngle = - np.float32(colList + 1 - (image_cols/2)) * ang_res_x + 90.0
    
    verticalAngle = verticalAngle / 180.0 * np.pi
    horizonAngle = horizonAngle / 180.0 * np.pi


    lengthList = thisImage.reshape(image_rows_full*image_cols)
    lengthList[lengthList > max_range] = 0.0
    lengthList[lengthList < min_range] = 0.0

    x = np.sin(horizonAngle) * np.cos(verticalAngle) * lengthList
    y = np.cos(horizonAngle) * np.cos(verticalAngle) * lengthList
    z = np.sin(verticalAngle) * lengthList + height
    
    points = np.column_stack((x,y,z))
    # delete points that has range value 0
    points = np.delete(points, np.where(lengthList==0), axis=0) # comment this line for visualize at the same speed (for video generation)

    colors = np.tile(color, (len(points), 1))
    pcd = trimesh.PointCloud(vertices = points, colors = colors)
    return pcd

flags = read_args()
SAVEDATA = flags.save_data
codd_path = flags.input_path
output_folder = flags.output_folder
mode = flags.mode
dataset =CODDAggDataset(codd_path, mode=mode, frame_rate = 25, maxDist=None)

if SAVEDATA:
    output_path = os.path.join(os.path.split(codd_path)[0], output_folder)
    if os.path.exists(output_path):
        pass
    else:
        print("Creat a New Folder for saving the output")
        os.mkdir(output_path)

    range_image_array = np.empty([0, image_rows_full, image_cols, 1], dtype=np.float32)
    intensity_map_array = np.empty([0, image_rows_full, image_cols, 1], dtype=np.float32)
    for data in dataset:
        pcd_1, pcd_2 = data[0]
        range_image_1, intensity_map_1 = create_range_map(pcd_1)
        range_image_2, intensity_map_2 = create_range_map(pcd_2)

        range_image_array = np.append(range_image_array, range_image_1, axis=0)
        range_image_array = np.append(range_image_array, range_image_2, axis=0)

        intensity_map_array = np.append(intensity_map_array, intensity_map_1, axis=0)
        intensity_map_array = np.append(intensity_map_array, intensity_map_2, axis=0)


    np.save(os.path.join(output_path, f"{mode}_range_map.npy"), range_image_array)
    np.save(os.path.join(output_path, f"{mode}_intensity_map.npy"), intensity_map_array)

    print('Dataset saved: {}'.format(mode))

else:
    p, t = dataset[0]
    pcd_1, pcd_2 = p

    # pcd_transform = np.concatenate((pcd_1[:, :3], np.ones((len(pcd_1), 1))), axis = 1)
    # pcd_transform = t @ pcd_transform.T
    # pcd_transform = pcd_transform.T[:, :3]
    # color_1 = np.tile([255, 0, 0], (len(pcd_transform), 1))
    # color_2 = np.tile([0, 0, 255], (len(pcd_2), 1))
    # pcd_all = trimesh.PointCloud(vertices = np.concatenate((pcd_transform, pcd_2[:, :3]), axis = 0), colors = np.concatenate((color_1, color_2), axis = 0))
    # pcd_all.export("vis_all_points.ply")

    # cNorm = colors.Normalize()
    # jet = plt.get_cmap('jet')
    # scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

    range_map, intensity_map = create_range_map(points_array=pcd_1)
    #range_map = scalarMap.to_rgba(range_map)
    recovered_pcd = recover_pcd(thisImage=range_map)

    # range_map = scalarMap.to_rgba(range_map)

    plt.imshow(range_map.reshape(image_rows_full, image_cols, -1))
    plt.savefig("range_map.png")
    plt.imshow(intensity_map.reshape(image_rows_full, image_cols, -1))
    plt.savefig("intensity_map.png")

    recovered_pcd.export("recover_40.ply")
    # pcd = trimesh.PointCloud(vertices = point_cloud_test[:,:3])

    # pcd.export("test.ply")
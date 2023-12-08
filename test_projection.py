import os
import numpy as np
import argparse
import math
import cv2
from durlar_utils.bin_to_img import *
import cv2
from PIL import Image, ImageOps
import trimesh

import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d




# offset_lut = np.array([63,42,21,1,61,41,22,2,60,41,22,3,59,41,22,4,59,40,22,4,58,40,23,5,57,40,23,6,57,40,23,6,56,40,23,6,56,39,23,7,56,39,23,7,55,39,23,7,55,39,23,7,55,39,23,7,55,39,23,7,55,39,23,7,55,39,23,7,55,39,23,7,55,39,23,7,55,39,23,7,55,39,23,7,55,39,23,7,56,39,23,6,56,39,23,6,56,40,23,6,57,40,22,5,57,40,22,5,58,40,22,4,58,40,22,3,59,41,22,2,60,41,21,1,61,41,21,0])

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

def get_low_res_from_high_res(high_res_data, image_rows_high = 128, upscaling_factor = 4):
    low_res_index = range(0, image_rows_high, upscaling_factor)
    low_res_data = high_res_data[:,low_res_index]
    return low_res_data


def upsample_image_bilinear(img_numpy, output_size):
    # Load the image
    # img = np.load(img_npy_path)

    # Resize the image
    upsampled_img = cv2.resize(img_numpy, output_size, interpolation = cv2.INTER_LINEAR)

    return upsampled_img


def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines

def writetxt(filename, file):
    """
    Write a list into a txt file
    """
    with open(filename, 'w') as fp:
        for item in file:
            # write each item on a new line
            fp.write("%s\n" % item)
    print('Done')

if __name__ == "__main__":
    # test_data = "/cluster/work/riner/users/biyang/dataset/depth_large_test.npy"
    # output_size = (128, 2048)

    # imgs = np.load(test_data)
    # imgs_low_res = get_low_res_from_high_res(imgs)

    # print(imgs_low_res.shape)


    # for img in imgs_low_res:
    #      print(img.shape)
    #      upsample_img = upsample_image_bilinear(img, output_size)
    #      print(upsample_img.shape)
    #      print(upsample_img.max())
    #      exit(0)



    # Test the KITTI dataset split
    train_split_file = "/cluster/work/riner/users/biyang/dataset/KITTI/train_files.txt"
    test_split_file = "/cluster/work/riner/users/biyang/dataset/KITTI/val_files.txt"


    train_split = readlines(train_split_file)
    val_split = readlines(test_split_file)



    print(len(train_split))

    # train_split_filtered = [name.split()[0] for name in train_split]
    # val_split_filtered = [name.split()[0] for name in val_split]

    

    # writetxt("/cluster/work/riner/users/biyang/dataset/KITTI/train_files.txt", train_split_filtered)
    # writetxt("/cluster/work/riner/users/biyang/dataset/KITTI/val_files.txt", val_split_filtered)


    exit()














    # scan = np.fromfile("/cluster/work/riner/users/biyang/dataset/DurLAR/DurLAR_20211209_S/ouster_points/data/0000000123.bin", dtype=np.float32)

    

    # scan = scan.reshape(-1, 4)
    	
    # rows = 128
    # cols = 2048

    
    # img_data = np.zeros((rows,cols))
    # img_range = np.zeros((rows,cols))
    # max_diff = -0.1
    # avg_err = 0
    # n_val = 0

    # for u in range(cols):
    #     for v in range(rows):

    #         idx = idx_from_px((u,v), cols)

    #         # Ouster has a kinda weird reprojection model, see page 12:
    #         # https://data.ouster.io/downloads/software-user-manual/software-user-manual-v2p0.pdf

    #         # Compensate beam to center offset
    #         xy_range = np.sqrt(scan[idx,0]**2 + scan[idx,1]**2) - origin_offset

    #         # Compensate beam to sensor bottom offset
    #         z = scan[idx,2] - lidar_to_sensor_z_offset

    #         # Calculate range as it's defined in the ouster manual
    #         img_range[v,u] = np.sqrt(xy_range**2 + z**2) + origin_offset

    #         # Reproject pixel with range to 3D point
    #         point_repro = px_to_xyz((u,v), img_range[v,u], cols)
    #         point_raw = [scan[idx,0], scan[idx,1], scan[idx,2]]

    #         # Check if point is valid
    #         if (img_range[v,u] > 0.1):
    #             p_diff = np.sqrt((point_repro[0]-scan[idx,0])**2 + (point_repro[1]-scan[idx,1])**2 + (point_repro[2]-scan[idx,2])**2)
    #             avg_err += p_diff
    #             n_val += 1
    #             if (p_diff > max_diff):
    #                 max_diff = p_diff
    #         img_data[v,u] = scan[idx,3]



    range_map = Image.open("/cluster/work/riner/users/biyang/dataset/DurLAR/DurLAR_20211209_S/depth/0000000250.png")
    gray_image = ImageOps.grayscale(range_map)
    gray_image = np.asarray(gray_image) 


    # Canny
    edges_canny = cv2.Canny(gray_image, threshold1=30, threshold2=100)

    # Sobel
    # sobelMagnitude = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=5)
    sobelMagnitude = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=5)

    # Compute the magnitude of the gradients
    # sobelMagnitude = np.sqrt(sobelx**2 + sobely**2)

    # Normalize the magnitude to range [0, 255]
    sobelMagnitude = (sobelMagnitude / np.max(sobelMagnitude) * 255).astype(np.uint8)


    cv2.imwrite('original.png', gray_image)
    cv2.imwrite('canny.png', edges_canny)
    cv2.imwrite('sobel.png', sobelMagnitude)

    exit(0)

    range_map = np.asarray(gray_image) / 255

    range_map_new = range_map + 0.1
    range_map_new[range_map == 0] = 0

    h, w = range_map.shape[:2]
    vis_row_index = range(0, h, 32)
    vis_row = range_map[vis_row_index, :]
    # vis_row = range_map[32:33, :]

    plt.figure()
    x = range(0, w)
    for i, row in enumerate(vis_row):
        # row_smoothed = gaussian_filter1d(row, sigma = 5)
        # plt.hist(row, bins=5, color='lightgray', edgecolor='black', label=f'hist{i}')
        # The first line
        plt.plot(x, row, label=f'row{i}')
        # plt.plot(x, row_smoothed, label=f'row_smoothed{i}')
        # Adding legend
        plt.legend()
    # plt.imshow()
    plt.savefig("test.png")

    exit(0)
         

    pcd_1 = img_to_pcd(range_map)
    pcd_2 = img_to_pcd(range_map_new)

    pcd_all = np.vstack((pcd_1, pcd_2))

    pcd_pred_color = np.zeros_like(pcd_1)
    pcd_pred_color[:, 0] = 255
    pcd_gt_color = np.zeros_like(pcd_2)
    pcd_gt_color[:, 2] = 255
    
    pcd_all_color = np.vstack((pcd_pred_color, pcd_gt_color))

    point_cloud = trimesh.PointCloud(
        vertices=pcd_all,
        colors=pcd_all_color)
    
    point_cloud.export("test.ply")     



    print(range_map.max())
    print(range_map_new.max())
    exit(0)
    
    # print("Average Error: ", avg_err/n_val)
    # print("Max Error: ", max_diff)


        

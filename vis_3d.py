import open3d as o3d
import numpy as np
from evaluation import img_to_pcd, img_to_pcd_carla, img_to_pcd_kitti
from datasets import rimg_loader, npy_loader, npy_loader_without_intensity

import cv2

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import json

from glob import glob

def colormap_img(img):
    cNorm = colors.Normalize(vmin=0, vmax=50)
    jet_loss_map = plt.get_cmap('jet')
    scalarMap_loss_map = cmx.ScalarMappable(norm=cNorm, cmap=jet_loss_map)


    img = scalarMap_loss_map.to_rgba(img)
    img = img[:,:3]

    return img



if __name__ == "__main__":
    path = r"D:\Documents\ETH\Master_Thesis\paper_vis\durlar\ours"

    ply_paths = glob(path + "/*.ply")


    for ply_file in ply_paths:

        pcd = o3d.io.read_point_cloud(ply_file)

        points = pcd.points
        points = np.asarray(points)

        if (ply_file.__contains__('carla')) and \
            (ply_file.split('/')[-1].__contains__("hat") or ply_file.__contains__("srno") or ply_file.split('/')[-1].__contains__("gt_low")):
                        
            print(ply_file)
            points[:, 2] = - points[:, 2]
            points[:, 0] = - points[:, 0]
            pcd.points = o3d.utility.Vector3dVector(points)
        
        distance = np.linalg.norm(points, axis=1)
        new_colors = colormap_img(distance)
        # elevation = points[:, 2]
        # new_colors = colormap_img(elevation)
        
        pcd.colors = o3d.utility.Vector3dVector(new_colors)


        o3d.io.write_point_cloud(ply_file, pcd)

        # o3d.visualization.draw_geometries([pcd])
        
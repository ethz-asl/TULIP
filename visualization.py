import open3d as o3d
import numpy as np
from evaluation import img_to_pcd, img_to_pcd_carla, img_to_pcd_kitti
from datasets import rimg_loader, npy_loader, npy_loader_without_intensity

import cv2

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import json

import trimesh


# cNorm = colors.Normalize(vmin=0, vmax=1)

def colormap_pcd(img):
    cNorm = colors.Normalize(vmin=0, vmax=80)
    jet_loss_map = plt.get_cmap('jet')
    scalarMap_loss_map = cmx.ScalarMappable(norm=cNorm, cmap=jet_loss_map)
    img = scalarMap_loss_map.to_rgba(img)
    img = img[:,:3]

    return img

def colormap_img(img):
    cNorm = colors.Normalize(vmin=img.min(), vmax=img.max())
    jet_loss_map = plt.get_cmap('jet')
    scalarMap_loss_map = cmx.ScalarMappable(norm=cNorm, cmap=jet_loss_map)


    img = scalarMap_loss_map.to_rgba(img)
    img = img[:, : ,:3]

    return img


def visulize_scores(results, metric = "Mean Absolute Error", skip = 25):
    # linestyles = ['-', '--', '-.', ':']
    linestyles = ['-']
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    label = ['Patch2x2',
             'Patch1x4 + Circular Padding',
             'Patch1x4 + Circular Padding + GridReshape',
             'Patch1x4 + Circular Padding + GridReshape + PixelShuffle',
             'Patch1x4 + Circular Padding + GridReshape + PixelShuffle + MC Dropout']

    # Create a figure and axis
    fig, ax = plt.subplots()
    x = np.arange(len(results[0]))

    skipping = range(0, len(x), skip)

    # Plot the data for all results using cycling linestyles and colors

    # ax.plot(x, results[0], label='sin(x)', color='blue')
    # # ax.plot(x, y2, label='cos(x)', linestyle='--', color='red')
    # # ax.plot(x, y3, label='tan(x)', linestyle='-.', color='green')
    # # ax.plot(x, y4, label='exp(x/10)', linestyle=':', color='purple')
    # # ax.plot(x, y5, label='log(x+1)', linestyle='-', color='orange')

    for i, y in enumerate(results):
        y = np.asarray(y)

        linestyle = linestyles[i % len(linestyles)]
        color = colors[i % len(colors)]
        ax.plot(x[skipping], y[skipping], linestyle=linestyle, color=color, label=label[i])

    # Set labels and title
    ax.set_xlabel('Image ID')
    # ax.set_ylabel(metric, fontsize=20)
    ax.set_title(metric, fontsize=15)

    # Add a legend
    ax.legend()

    # Show the plot
    plt.show()




if __name__ == "__main__":


    file_path = r"C:\Users\ybMas\Downloads\results_mcdrop.txt"
    with open(file_path, 'r') as file:
        loaded_dict = json.load(file)

    mae = np.array(loaded_dict["mae"])
    iou = np.array(loaded_dict["iou"])
    chamfer_dist = np.array(loaded_dict["chamfer_dist"])


    top20_mae = np.argsort(mae)[0:50]
    top20_chamfer_dist = np.argsort(chamfer_dist)[0:200]
    top20_iou = np.argsort(iou)[::-1][0:200]




    # print(top20_chamfer_dist,top20_iou)

    intersect1 = np.intersect1d(top20_chamfer_dist, top20_iou)
    # intersect2 = np.intersect1d(intersect1, top20_iou)

    print(intersect1)
    print(mae[intersect1], iou[intersect1], chamfer_dist[intersect1])
    # index = [496,  605,  702,  926,  955, 1029, 1046, 1120, 1271, 1640, 1689, 1705, 1783, 1844 ,1875, 2440]

    # files = [r"C:\Users\ybMas\Downloads\1323.rimg",
    #          r"C:\Users\ybMas\Downloads\500.rimg",]
    #         #  r"C:\Users\ybMas\Downloads\0000001058.npy",
    #         #  r"C:\Users\ybMas\Downloads\0000001054.npy"]
    

    # low_index_1 = range(0, 128, 4)
    # low_index_2 = range(1, 129, 4)
    # low_index_3 = range(2, 130, 4)
    # low_index_4 = range(3, 131, 4)






    # for file in files:

    #     high_range_map = rimg_loader(file)
    #     low_range_map = high_range_map[low_index_1, :]

    #     high_points = img_to_pcd_carla(high_range_map / 80)
    #     low_points = img_to_pcd_carla(low_range_map / 80)


    #     high_pcd = trimesh.points.PointCloud(high_points)
    #     low_pcd = trimesh.points.PointCloud(low_points)


    #     high_pcd.colors = colormap_pcd(np.linalg.norm(high_points, axis=1))
    #     low_pcd.colors = colormap_pcd(np.linalg.norm(low_points, axis=1))




        
    #     # high_pcd.export(file.replace(".rimg", ".ply"))
    #     # low_pcd.export(file.replace(".rimg", "_low.ply"))
    #     # cv2.imwrite(file.replace(".rimg", ".png"), high_range_map / 80*255)
    #     # cv2.imwrite(file.replace(".rimg", "_low.png"), low_range_map/80*255)

    #     # continue

    #     low_range_map_1 = high_range_map[low_index_1, :]
    #     low_range_map_2 = high_range_map[low_index_2, :]
    #     low_range_map_3 = high_range_map[low_index_3, :]
    #     low_range_map_4 = high_range_map[low_index_4, :]


    #     delta_1 = low_range_map_1 - low_range_map_2
    #     delta_2 = low_range_map_1 - low_range_map_3
    #     delta_3 = low_range_map_1 - low_range_map_4

    #     print(delta_1.mean(), delta_2.mean(), delta_3.mean())

    #     print(np.std(delta_1), np.std(delta_2), np.std(delta_3))
    #     plt.hist(delta_1.reshape(-1), bins=20, edgecolor='k', range = (-0.002, 0.002))
    #     plt.show()
    #     plt.close()

    #     plt.hist(delta_2.reshape(-1), bins=20, edgecolor='k', range = (-0.002, 0.002))
    #     plt.show()
    #     plt.close()

    #     plt.hist(delta_3.reshape(-1), bins=20, edgecolor='k', range = (-0.002, 0.002))
    #     plt.show()
    #     plt.close()


        

             



    # file_paths = [r"D:\Documents\ETH\Master_Thesis\data\kitti\results_patch2x2.txt",
    #                 r"D:\Documents\ETH\Master_Thesis\data\kitti\results_logtransform_patch1x4.txt",
    #                 r"D:\Documents\ETH\Master_Thesis\data\kitti\results_logtransform_patch1x4_gridreshape_circularpadding.txt",
    #                 r"D:\Documents\ETH\Master_Thesis\data\kitti\results_nopretrain.txt",
    #                 r"D:\Documents\ETH\Master_Thesis\data\kitti\results_mcdrop_nopretrain.txt",]

    # mae = []
    # chamfer_dist = []
    # iou = []
    # precision = []
    # recall = []
    # for file_path in file_paths:
    #     with open(file_path, 'r') as file:
    #         loaded_dict = json.load(file)

    #     mae.append(loaded_dict["mae"])
    #     chamfer_dist.append(loaded_dict["chamfer_dist"])
    #     iou.append(loaded_dict["iou"])
    #     precision.append(loaded_dict["precision"])
    #     recall.append(loaded_dict["recall"])


    # visulize_scores(mae, metric = "Mean Absolute Error")
    # visulize_scores(chamfer_dist, metric = "Chamfer Distance")
    # visulize_scores(iou, metric = "IoU")
    # visulize_scores(precision, metric = "Precision")
    # visulize_scores(recall, metric = "Recall")




    # carla_data = rimg_loader(r"D:\Documents\ETH\Master_Thesis\data\carla\1269.rimg")

    # durlar_data = npy_loader(r"D:\Documents\ETH\Master_Thesis\data\durlar\0000001019.npy")
    # # durlar_data = durlar_data * 120

    # kitti_data = npy_loader_without_intensity(r"D:\Documents\ETH\Master_Thesis\data\kitti\00000013.npy")


    # # carla_img = colormap_img(carla_data)

    # # cv2.imwrite("carla_img.png", carla_img*255)
    # # cv2.imwrite("durlar_img.png", colormap_img(durlar_data)*255)
    # cv2.imwrite("kitti_img.png", colormap_img(np.flip(kitti_data))*255)

    # exit(0)






    # points_durlar = img_to_pcd(durlar_data)
    # pcd_durlar = o3d.geometry.PointCloud()
    # pcd_durlar.points = o3d.utility.Vector3dVector(points_durlar)



    # points_carla = img_to_pcd_carla(carla_data / 80)
    # pcd_carla = o3d.geometry.PointCloud()
    # pcd_carla.points = o3d.utility.Vector3dVector(points_carla)
    # o3d.visualization.draw_geometries([pcd_carla])


    # points_kitti = img_to_pcd_kitti(kitti_data / 120)
    # pcd_kitti = o3d.geometry.PointCloud()
    # pcd_kitti.points = o3d.utility.Vector3dVector(points_kitti)
    # o3d.visualization.draw_geometries([pcd_kitti])







    


    
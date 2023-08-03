
import wandb
import matplotlib.pyplot as plt
import trimesh
from mae.util.evaluation import *
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx

test_data_path = "/cluster/work/riner/users/biyang/dataset/depth_large_test.npy"
output_size = (128, 2048)
upscaling_factor = 4
imgae_rows_high = output_size[0]

output_dir = "/cluster/work/riner/users/biyang/experiment/"
experiment_name = "Bilinear"
root_dir = os.path.join(output_dir, experiment_name)
if not os.path.exists(root_dir):
    os.mkdir(root_dir)

pcd_eval_path = os.path.join(root_dir, "pcd")
if not os.path.exists(pcd_eval_path):
    os.mkdir(pcd_eval_path)


wandb_disabled = False
save_pcd = True
# project_name = "lidar_sr_network"
project_name = "experiment_upsampling"
entity = "biyang"
run_name = "bilinear_upsampling"

cNorm = colors.Normalize(vmin=0, vmax=1)
jet = plt.get_cmap('viridis_r')
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)


if wandb_disabled:
    mode = "disabled"
else:
    mode = "online"
wandb.init(project=project_name,
            entity=entity,
            name = run_name, 
            mode=mode,)

    


def upsample_image_bilinear(img_numpy, output_size):
    # Load the image
    # img = np.load(img_npy_path)

    # Resize the image
    upsampled_img = cv2.resize(img_numpy.transpose(), output_size, interpolation = cv2.INTER_LINEAR)
    # print(upsampled_img.shape)
    # exit(0)

    upsampled_img = upsampled_img.transpose()
    
    return upsampled_img


def get_low_res_from_high_res(high_res_data, image_rows_high = 128, upscaling_factor = 4):
    low_res_index = range(0, image_rows_high, upscaling_factor)
    low_res_data = high_res_data[:,low_res_index]
    return low_res_data



def main():
    imgs = np.load(test_data_path)
    imgs_low_res = get_low_res_from_high_res(imgs, image_rows_high=imgae_rows_high, upscaling_factor=upscaling_factor)


    grid_size = 0.1
    local_step = 0


    for img_low_res, img_high_res in zip(imgs_low_res, imgs):
        pred = upsample_image_bilinear(img_low_res, output_size)
        gt = img_high_res


        mse_all = ((pred - gt)**2).mean()
        pcd_pred = img_to_pcd(pred)
        pcd_gt = img_to_pcd(gt)

        # print(pcd_pred.shape, pcd_gt.shape)

        pcd_all = np.vstack((pcd_pred, pcd_gt))

        chamfer_dist = chamfer_distance(pcd_gt, pcd_pred)
        min_coord = np.min(pcd_all, axis=0)
        max_coord = np.max(pcd_all, axis=0)
        
        # print("chamfer_dist: ", chamfer_dist)
        # Voxelize the ground truth and prediction point clouds
        voxel_grid_predicted = voxelize_point_cloud(pcd_pred, grid_size, min_coord, max_coord)
        voxel_grid_ground_truth = voxelize_point_cloud(pcd_gt, grid_size, min_coord, max_coord)
        # print("Voxelize")
        # print(voxel_grid_ground_truth.shape, voxel_grid_predicted.shape)
        # Calculate metrics
        iou, precision, recall = calculate_metrics(voxel_grid_predicted, voxel_grid_ground_truth)
        # print("Evaluation")
        # print(iou, precision, recall)

        if save_pcd:
            
            if not os.path.exists(pcd_eval_path):
                os.mkdir(pcd_eval_path)
            pcd_pred_color = np.zeros_like(pcd_pred)
            pcd_pred_color[:, 0] = 255
            pcd_gt_color = np.zeros_like(pcd_gt)
            pcd_gt_color[:, 2] = 255
            
            pcd_all_color = np.vstack((pcd_pred_color, pcd_gt_color))

            point_cloud = trimesh.PointCloud(
                vertices=pcd_all,
                colors=pcd_all_color)
            
            point_cloud.export(os.path.join(pcd_eval_path, f"pred_gt_{local_step}.ply"))

            

        wandb.log({"Test/mse_all": mse_all, 
                   "Test/mse_low_res": 0,
                   "Test/chamfer_dist": chamfer_dist, 
                   "Test/iou": iou,
                   "Test/precision": precision,
                   "Test/recall": recall}, local_step)

        gt_vis = scalarMap.to_rgba(gt)[..., :3]
        pred_vis = scalarMap.to_rgba(pred)[..., :3]

        f, ax = plt.subplots(2,1, figsize=(16, 2))
        ax[0].imshow(gt_vis, interpolation='none', aspect="auto")
        ax[0].axis("off")
        ax[1].imshow(pred_vis, interpolation='none', aspect="auto")
        ax[1].axis("off")
        # plt.suptitle("gt-prediction")
        plt.subplots_adjust(wspace=.05, hspace=.05)
        logger = wandb.Image(f)
        plt.close()      
        wandb.log({"gt - pred":
                logger})
        
        
        local_step += 1
        
         

if __name__ == "__main__":
    main()
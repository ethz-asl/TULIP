
import wandb
import matplotlib.pyplot as plt
import trimesh
from mae.util.evaluation import *
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx

from mae.util.datasets import build_durlar_upsampling_dataset, build_carla_upsampling_dataset, build_kitti_upsampling_dataset, build_carla200000_upsampling_dataset
import tqdm

import json

def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--dataset_select', default='durlar', type=str, choices=['durlar', 'carla','kitti', 'image-net', 'carla200000'])

    parser.add_argument('--gray_scale', action="store_true", help='use gray scale imgae')
    parser.add_argument('--img_size_low_res', nargs="+", type=int, help='low resolution image size, given in format h w')
    parser.add_argument('--img_size_high_res', nargs="+", type=int, help='high resolution image size, given in format h w')
    # parser.add_argument('--in_chans', type=int, default = 1, help='number of channels')
    parser.add_argument('--data_path_low_res', default=None, type=str,
                        help='low resolution dataset path')
    parser.add_argument('--data_path_high_res', default=None, type=str,
                        help='high resolution dataset path')
    
    parser.add_argument('--save_pcd', action="store_true", help='save pcd output in evaluation step')
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    
    parser.add_argument('--wandb_disabled', action='store_true', help="disable wandb")
    parser.add_argument('--entity', type = str, default = "biyang")
    parser.add_argument('--project_name', type = str, default = "Ouster_MAE")
    parser.add_argument('--run_name', type = str, default = None)




    parser.add_argument('--crop', action="store_true", help='crop the image to 128 x 128 (default)')
    parser.add_argument('--mask_loss', action="store_true", help='Mask the loss value with no LiDAR return')
    parser.add_argument('--use_intensity', action="store_true", help='use the intensity as the second channel')
    parser.add_argument('--reverse_pixel_value', action="store_true", help='reverse the pixel value in the input')
    parser.add_argument('--log_transform', action="store_true", help='apply log1p transform to data')
    parser.add_argument('--keep_close_scan', action="store_true", help='mask out pixel belonging to further object')
    parser.add_argument('--keep_far_scan', action="store_true", help='mask out pixel belonging to close object')
    


    return parser



# test_data_path = "/cluster/work/riner/users/biyang/dataset/depth_large_test.npy"
# test_data_path = "/cluster/work/riner/users/biyang/dataset/depth_intensity_new/val/depth/all"
# output_size = (128, 2048)
# upscaling_factor = 4
# imgae_rows_high = output_size[0]

# output_dir = "/cluster/work/riner/users/biyang/experiment/"
# experiment_name = "Bilinear"
# root_dir = os.path.join(output_dir, experiment_name)
# if not os.path.exists(root_dir):
#     os.mkdir(root_dir)

# pcd_eval_path = os.path.join(root_dir, "pcd")
# if not os.path.exists(pcd_eval_path):
#     os.mkdir(pcd_eval_path)


# wandb_disabled = False
# save_pcd = True
# # project_name = "lidar_sr_network"
# project_name = "durlar_evaluation"
# entity = "biyang"
# run_name = "bilinear"

cNorm = colors.Normalize(vmin=0, vmax=1)
jet = plt.get_cmap('viridis_r')
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)


# if wandb_disabled:
#     mode = "disabled"
# else:
#     mode = "online"
# wandb.init(project=project_name,
#             entity=entity,
#             name = run_name, 
#             mode=mode,)

    


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



def main(args):
    if args.wandb_disabled:
        mode = "disabled"
    else:
        mode = "online"
    wandb.init(project=args.project_name,
                entity=args.entity,
                name = args.run_name, 
                mode=mode,)
    

    if args.dataset_select == 'durlar':
        indices = [481, 496, 860, 869, 894, 1482, 1491, 1783, 2010]
        dataset_val = build_durlar_upsampling_dataset(is_train = False, args = args)
    elif args.dataset_select == 'carla':
        dataset_val = build_carla_upsampling_dataset(is_train = False, args = args)
        indices = [57  , 68 ,  79 , 101 , 113 , 124  ,215 , 315, 354, 387,388,456 ,542,564 ,815,816  ,870 ,883, 1018 ,1030 ,1057 , 1085 ,1159 ,1172 , 1371, 1500]
    elif args.dataset_select == 'kitti':
        indices = [188, 2054,496, 926 , 979,1120,1875, 666, 999, 888, 520, 1314, 2000, 688, 1, 10, 100, 777, 1111, 2222, 369, 796, 1000]
        dataset_val = build_kitti_upsampling_dataset(is_train = False, args = args)
    else:
        raise NotImplementedError("Cannot find the matched dataset builder")
    

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, 
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    h_low_res = tuple(args.img_size_low_res)[0]
    h_high_res = tuple(args.img_size_high_res)[0]
    downsampling_factor = h_high_res // h_low_res

    grid_size = 0.1
    local_step = 0
    global_step = 0
    total_loss = 0
    total_iou = 0
    total_cd = 0
    local_step = 0

    # evaluation_metrics = {'mae':[],
    #                       'chamfer_dist':[],
    #                       'iou':[],
    #                       'precision':[],
    #                       'recall':[]}



    for batch in tqdm.tqdm(data_loader_val):

        if indices is not None:
            if global_step not in indices:
                global_step += 1
                continue

        # start_time = time.time()

        images_low_res = batch[0][0] # (B=1, C, H, W)
        images_high_res = batch[1][0] # (B=1, C, H, W)

        images_high_res = images_high_res.permute(0, 2, 3, 1).squeeze()
        images_low_res = images_low_res.permute(0, 2, 3, 1).squeeze()
        

        images_high_res = images_high_res.detach().cpu().numpy()
        images_low_res = images_low_res.detach().cpu().numpy()



        pred_img = upsample_image_bilinear(images_low_res, output_size = tuple(args.img_size_high_res))

        global_step += 1

        mae_all = (np.abs(pred_img - images_high_res)).mean()
        if args.dataset_select in ["carla", "carla200000"]:
            # Carla has different projection process as durlar
            # Refer to code in iln github
            pred_img = np.flip(pred_img)
            images_high_res = np.flip(images_high_res)

            pcd_pred = img_to_pcd_carla(pred_img, maximum_range = 80)
            pcd_gt = img_to_pcd_carla(images_high_res, maximum_range = 80)
        
        elif args.dataset_select == "kitti":
            # 3D Evaluation Metrics
            pcd_pred = img_to_pcd_kitti(pred_img, maximum_range= 120)
            pcd_gt = img_to_pcd_kitti(images_high_res, maximum_range = 120)


        elif args.dataset_select == "durlar":

            # 3D Evaluation Metrics
            pcd_pred = img_to_pcd(pred_img, maximum_range= 120)
            pcd_gt = img_to_pcd(images_high_res, maximum_range = 120)

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

        # evaluation_metrics['mae'].append(mae_all)
        # evaluation_metrics['chamfer_dist'].append(chamfer_dist.item())
        # evaluation_metrics['iou'].append(iou)
        # evaluation_metrics['precision'].append(precision)
        # evaluation_metrics['recall'].append(recall)

        if global_step % 100 == 0 or global_step == 1 or indices is not None:

            wandb.log({"Test/mse_all": mae_all, 
                   "Test/mse_low_res": 0,
                   "Test/chamfer_dist": chamfer_dist, 
                   "Test/iou": iou,
                   "Test/precision": precision,
                   "Test/recall": recall}, local_step)

            gt_vis = scalarMap.to_rgba(images_high_res)[..., :3]
            pred_vis = scalarMap.to_rgba(pred_img)[..., :3]

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
            if args.save_pcd:
                if local_step % 4 == 0 or indices is not None:
                    pcd_outputpath = os.path.join(args.output_dir, 'pcd') if indices is None else os.path.join("/cluster/home/biyang", 'pcd_vispaper')
                    if not os.path.exists(pcd_outputpath):
                        os.mkdir(pcd_outputpath)
                    pcd_pred_color = np.zeros_like(pcd_pred)
                    pcd_pred_color[:, 0] = 255
                    pcd_gt_color = np.zeros_like(pcd_gt)
                    pcd_gt_color[:, 2] = 255
                    
                    # pcd_all_color = np.vstack((pcd_pred_color, pcd_gt_color))

                    point_cloud_pred = trimesh.PointCloud(
                        vertices=pcd_pred,
                        colors=pcd_pred_color)
                    
                    point_cloud_gt = trimesh.PointCloud(
                        vertices=pcd_gt,
                        colors=pcd_gt_color)
                    
                    point_cloud_pred.export(os.path.join(pcd_outputpath, f"pred_{global_step}.ply"))  
                    point_cloud_gt.export(os.path.join(pcd_outputpath, f"gt_{global_step}.ply"))    
            

        
        
        
            local_step += 1

        total_iou += iou
        total_cd += chamfer_dist
        total_loss += mae_all

    # evaluation_file_path = os.path.join(args.output_dir,'results.txt')
    # with open(evaluation_file_path, 'w') as file:
    #     json.dump(evaluation_metrics, file)

    # print(print(f'Dictionary saved to {evaluation_file_path}'))

    wandb.log({'Metrics/test_average_iou': total_iou/global_step,
                'Metrics/test_average_cd': total_cd/global_step,
                'Metrics/test_average_loss': total_loss/global_step})
        
         

if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    main(args=args)
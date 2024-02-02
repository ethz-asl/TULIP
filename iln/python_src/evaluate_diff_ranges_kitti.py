import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Datasets
from dataset.samples_from_image_dataset import SamplesFromImageDataset
from dataset.dataset_utils import generate_dataset, denormalization_ranges

# Models
from models.iln.iln import ILN
from models.liif_cvpr21.liif_lidar import LIIFLiDAR
from models.interpolation.interpolation import Interpolation
from models.lsr_ras20.unet import UNet
from models.model_utils import generate_model

# Metric
# from metric.mae_evaluator import MAEEvaluator
# from metric.voxel_iou_evaluator import VoxelIoUEvaluator
import yaml
from mae.util.evaluation import * 
from mae.util.datasets import CropRanges
import wandb
import trimesh
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx

import json

cNorm = colors.Normalize(vmin=0, vmax=1)
jet = plt.get_cmap('viridis_r')
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)


def test_pixel_based_network(output_path, pred_batch=1, h_high = 64, w_high = 1024, save_pcd = True, grid_size = 0.1):
    # mae_evaluator = MAEEvaluator()
    # voxel_evaluator = VoxelIoUEvaluator(voxel_size=args.voxel_size, lidar=test_dataset.lidar_out)

    global_step = 0

    iteration_crop_ranges = 8
    max_range = 120

    total_loss = np.zeros(iteration_crop_ranges)
    total_iou = np.zeros(iteration_crop_ranges)
    total_cd = np.zeros(iteration_crop_ranges)
    total_f1 = np.zeros(iteration_crop_ranges)
    total_precision = np.zeros(iteration_crop_ranges)
    total_recall = np.zeros(iteration_crop_ranges)
    total_frames = np.zeros(iteration_crop_ranges)

    evaluation_metrics = {'mae':[],
                          'chamfer_dist':[],
                          'iou':[],
                          'precision':[],
                          'recall':[],
                          'f1':[]}



    for packed_batches in tqdm(test_loader, leave=False, desc='test'):

        # input_range_image:    [N, 1, H_in, W_in]
        # output_ranges:        [N, H_out*W_out, 1]
        input_range_image, output_ranges = packed_batches[0].cuda(), packed_batches[1].cuda()

        # Prediction: input_range_image [N, 1, H_in, W_in] --> pred_ranges: [N, H_out*W_out, 1]
        with torch.no_grad():
            # [-1 ~ 1] -> [ 0 ~ 1]
            input_range_image += 1.0
            input_range_image *= 0.5
            pred_ranges = model(input_range_image).view(output_ranges.shape[0], -1, 1)
            # [ 0 ~ 1] -> [-1 ~ 1]
            pred_ranges *= 2.0
            pred_ranges -= 1.0

        # Evaluations
        output_ranges = denormalization_ranges(output_ranges, norm_r =  test_dataset.lidar_out['norm_r'])  # [N * H_out * W_out]
        pred_ranges = denormalization_ranges(pred_ranges, norm_r =  test_dataset.lidar_out['norm_r'])  # [N * H_out * W_out]
        pred_ranges[pred_ranges < 0.] = 0.
        pred_ranges[pred_ranges > test_dataset.lidar_out['norm_r']] = test_dataset.lidar_out['norm_r']

        # MAE
        pred_ranges[pred_ranges < 0.] = 0.
        # pred_ranges[pred_ranges > test_dataset.lidar_out['norm_r']] = test_dataset.lidar_out['norm_r']
        pred_ranges[pred_ranges > test_dataset.lidar_out['max_r']] = 0
        output_ranges[output_ranges > test_dataset.lidar_out['max_r']] = 0

        pred_ranges = pred_ranges / test_dataset.lidar_out['norm_r']
        output_ranges = output_ranges / test_dataset.lidar_out['norm_r']
        # Reshape to [H, W]

        pred_ranges = pred_ranges.reshape(h_high, w_high)
        output_ranges = output_ranges.reshape(h_high, w_high)

        pred_full = pred_ranges.clone()
        gt_full = output_ranges.clone()

        
        for i in range(iteration_crop_ranges):
            min_dist = i*10 / max_range
            max_dist = (i+1)*10 / max_range
            range_cropper = CropRanges(min_dist, max_dist)

            pred,_= range_cropper(pred_full)
            gt, num_pixels = range_cropper(gt_full)

            if num_pixels != 0:
                total_frames[i] += 1
            else:
                continue

            pred = pred.detach().cpu().numpy()
            gt = gt.detach().cpu().numpy()


            low_res_index = range(0, h_high, 4)

            # Evaluate the loss of low resolution part
            loss_low_res_part = np.abs((pred[low_res_index, :] - gt[low_res_index, :]))
            loss_low_res_part = loss_low_res_part.mean()

            mse_all = (np.absolute(pred - gt)).sum() / num_pixels

            pcd_pred = img_to_pcd_kitti(pred, maximum_range=test_dataset.lidar_out['norm_r'])
            pcd_gt = img_to_pcd_kitti(gt, maximum_range=test_dataset.lidar_out['norm_r'])


            pcd_all = np.vstack((pcd_pred, pcd_gt))

            chamfer_dist = chamfer_distance(pcd_gt, pcd_pred, num_points=num_pixels)
            min_coord = np.min(pcd_all, axis=0)
            max_coord = np.max(pcd_all, axis=0)

            
            voxel_grid_predicted = voxelize_point_cloud(pcd_pred, grid_size, min_coord, max_coord)
            voxel_grid_ground_truth = voxelize_point_cloud(pcd_gt, grid_size, min_coord, max_coord)

            iou, precision, recall = calculate_metrics(voxel_grid_predicted, voxel_grid_ground_truth)

            f1 = 2 * (precision * recall) / (precision + recall)

            total_iou[i] += iou
            total_cd[i] += chamfer_dist
            total_loss[i] += mse_all
            total_f1[i] += f1
            total_precision[i] += precision
            total_recall[i] += recall

        global_step += 1

    evaluation_metrics['mae'] = list(total_loss / total_frames)
    evaluation_metrics['chamfer_dist'] = list(total_cd / total_frames)
    evaluation_metrics['iou'] = list(total_iou / total_frames)
    evaluation_metrics['precision'] = list(total_precision / total_frames)
    evaluation_metrics['recall'] = list(total_recall / total_frames)
    evaluation_metrics['f1'] = list(total_f1 / total_frames)

    evaluation_file_path = os.path.join(output_path,'evaluate_different_ranges_mcdrop.txt')
    with open(evaluation_file_path, 'w') as file:
        json.dump(evaluation_metrics, file) 

    print(print(f'Dictionary saved to {evaluation_file_path}'))
                
                
            



def test_implicit_network(output_path, pred_batch=1, h_high = 64, w_high = 1024, save_pcd = True, grid_size = 0.1):
    # mae_evaluator = MAEEvaluator()
    # voxel_evaluator = VoxelIoUEvaluator(voxel_size=args.voxel_size, lidar=test_dataset.lidar_out)
    global_step = 0

    iteration_crop_ranges = 8
    max_range = 120

    total_loss = np.zeros(iteration_crop_ranges)
    total_iou = np.zeros(iteration_crop_ranges)
    total_cd = np.zeros(iteration_crop_ranges)
    total_f1 = np.zeros(iteration_crop_ranges)
    total_precision = np.zeros(iteration_crop_ranges)
    total_recall = np.zeros(iteration_crop_ranges)
    total_frames = np.zeros(iteration_crop_ranges)

    evaluation_metrics = {'mae':[],
                          'chamfer_dist':[],
                          'iou':[],
                          'precision':[],
                          'recall':[],
                          'f1':[]}
    for packed_batches in tqdm(test_loader, leave=False, desc='test'):

        # if global_step > 10:
        #     break

        
        # input_range_image:    [N, 1, H_in, W_in]
        # input_queries:        [N, H_out*W_out, 2]
        # output_ranges:        [N, H_out*W_out, 1]
        input_range_image, input_queries, output_ranges = packed_batches[0].cuda(), packed_batches[1].cuda(), packed_batches[2].cuda()

        # Prediction: input_range_image [N, 1, H_in, W_in] --> pred_ranges: [N, H_out*W_out, 1]
        if input_range_image.shape[0] == 1 and pred_batch > 1:
            with torch.no_grad():
                input_range_image = input_range_image.repeat(num_of_gpus, 1, 1, 1)   # for multi-gpus
                input_queries = input_queries.view(num_of_gpus * pred_batch, -1, 2)  #
                preds = []
                for n in range(pred_batch):
                    bs = n * num_of_gpus
                    be = (n + 1) * num_of_gpus
                    preds.append(model(input_range_image, input_queries[bs:be, :, :]).view(1, -1, 1))
                pred_ranges = torch.cat(preds, dim=1).view(1, -1, 1)
        else:
            with torch.no_grad():
                pred_ranges = model(input_range_image, input_queries)

        # Evaluations
        output_ranges = denormalization_ranges(output_ranges, norm_r =  test_dataset.lidar_out['norm_r'])   # [N * H_out * W_out]
        pred_ranges = denormalization_ranges(pred_ranges, norm_r =  test_dataset.lidar_out['norm_r'])       # [N * H_out * W_out]
        pred_ranges[pred_ranges < 0.] = 0.

        pred_ranges[pred_ranges > test_dataset.lidar_out['max_r']] = 0
        output_ranges[output_ranges > test_dataset.lidar_out['max_r']] = 0

        pred_ranges = pred_ranges / test_dataset.lidar_out['norm_r']
        output_ranges = output_ranges / test_dataset.lidar_out['norm_r']
        # Reshape to [H, W]

        pred_ranges = pred_ranges.reshape(h_high, w_high)
        output_ranges = output_ranges.reshape(h_high, w_high)

       

        pred_full = pred_ranges.clone()
        gt_full = output_ranges.clone()

        for i in range(iteration_crop_ranges):
            min_dist = i*10 / max_range
            max_dist = (i+1)*10 / max_range
            range_cropper = CropRanges(min_dist, max_dist)

            pred,_= range_cropper(pred_full)
            gt, num_pixels = range_cropper(gt_full)

            pred = pred.detach().cpu().numpy()
            gt = gt.detach().cpu().numpy()

            if num_pixels != 0:
                total_frames[i] += 1
            else:
                continue

            low_res_index = range(0, h_high, 4)


            # Evaluate the loss of low resolution part
            loss_low_res_part = np.abs((pred[low_res_index, :] - gt[low_res_index, :]))
            loss_low_res_part = loss_low_res_part.mean()

            mse_all = (np.absolute(pred - gt)).sum() / num_pixels

            pcd_pred = img_to_pcd_kitti(pred, maximum_range=120)
            pcd_gt = img_to_pcd_kitti(gt, maximum_range=120)

            pcd_all = np.vstack((pcd_pred, pcd_gt))

            chamfer_dist = chamfer_distance(pcd_gt, pcd_pred, num_points=num_pixels)
            min_coord = np.min(pcd_all, axis=0)
            max_coord = np.max(pcd_all, axis=0)

            
            voxel_grid_predicted = voxelize_point_cloud(pcd_pred, grid_size, min_coord, max_coord)
            voxel_grid_ground_truth = voxelize_point_cloud(pcd_gt, grid_size, min_coord, max_coord)

            iou, precision, recall = calculate_metrics(voxel_grid_predicted, voxel_grid_ground_truth)


            f1 = 2 * (precision * recall) / (precision + recall)

            total_iou[i] += iou
            total_cd[i] += chamfer_dist
            total_loss[i] += mse_all
            total_f1[i] += f1
            total_precision[i] += precision
            total_recall[i] += recall

        global_step += 1

    evaluation_metrics['mae'] = list(total_loss / total_frames)
    evaluation_metrics['chamfer_dist'] = list(total_cd / total_frames)
    evaluation_metrics['iou'] = list(total_iou / total_frames)
    evaluation_metrics['precision'] = list(total_precision / total_frames)
    evaluation_metrics['recall'] = list(total_recall / total_frames)
    evaluation_metrics['f1'] = list(total_f1 / total_frames)

    evaluation_file_path = os.path.join(output_path,'evaluate_different_ranges_mcdrop.txt')
    with open(evaluation_file_path, 'w') as file:
        json.dump(evaluation_metrics, file) 

    print(print(f'Dictionary saved to {evaluation_file_path}'))
                
                


if __name__ == '__main__':
    # Parse the arguments
    parser = argparse.ArgumentParser(description="Evaluate the MAE (Mean Absolute Error) on the Carla dataset")
    parser.add_argument('-c', '--config',
                        type=str,
                        required=True,
                        help='Configuration filename. [.yaml]')
    parser.add_argument('-o', '--output_directory',
                        type=str,
                        required=False,
                        default=None,
                        help='Output directory')
    parser.add_argument('-b', '--batch',
                        type=int,
                        required=False,
                        default=7,
                        choices=[1, 2, 7, 11, 14, 17, 22, 34],
                        help='Batch size for network testing. (default: 7)')
    parser.add_argument('-cp', '--checkpoint',
                        type=str,
                        required=True,
                        help='Check point filename. [.pth]')
    parser.add_argument('-v', '--voxel_size',
                        type=float,
                        required=False,
                        default=0.1,
                        help='Voxel size for visualization, Default: 0.1')
    args = parser.parse_args()

    # Load the check point
    check_point = torch.load(args.checkpoint)




    # Load the configurations
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    dataset_config = config['dataset']
    # Generate dataset
    batch_size = args.batch
    test_dataset = generate_dataset(dataset_config)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    # Model
    model = generate_model(check_point['model']['name'], check_point['model']['args'])
    model.load_state_dict(check_point['model']['state_dict'])
    num_of_gpus = torch.cuda.device_count()
    if torch.cuda.device_count() > 1:
        model = nn.parallel.DataParallel(model)

                # group="DDP" if num_of_gpus > 1 else "Single_GPU",)

    model.eval().cuda()

    # Output directory
    output_directory = args.output_directory if args.output_directory is not None else os.path.dirname(args.checkpoint)
    # eval_result_filename = os.path.splitext(os.path.basename(args.checkpoint))[0]
    # eval_result_filename = os.path.join(output_directory, eval_result_filename + '[' + dataset_config['args']['res_out'] + ']')

    print("======================= Configuration ========================  ")
    model_name = check_point['model']['name']
    model_directory = args.checkpoint
    print('  Model:', model_name,
          '(' + str(sum(p.numel() for p in model.parameters() if p.requires_grad)) + ' parameters)')
    for key, value in check_point['model']['args'].items():
        print('    ' + key + ':', value)
    print('  Check point file:', args.checkpoint)
    print('  ')
    print('  Dataset: Carla', '(' + str(len(test_dataset)) + ' pairs)')
    print('  Batch:', batch_size)
    # print('  Output filename:', eval_result_filename)
    print('  Target resolution:', test_dataset.lidar_out['channels'], 'x', test_dataset.lidar_out['points_per_ring'])
    print("==============================================================  \n")

    # Evaluate the network
    if check_point['model']['name'].find('lsr') != -1:
        test_pixel_based_network(output_path = args.output_directory, save_pcd = config['logger']['save_pcd'], grid_size=args.voxel_size)
    else:
        test_implicit_network(output_path = args.output_directory, save_pcd = config['logger']['save_pcd'], grid_size=args.voxel_size)

    # Report the evaluation result
    # if not os.path.isfile(eval_result_filename):
    #     eval_result_file = open(eval_result_filename, 'w')
    #     eval_result_file.write('epoch |    mae   |    iou   |    pre   |    rec   |    f1    \n')   # HEADER
    #     eval_result_file.close()

    # eval_result_file = open(eval_result_filename, 'a')
    # eval_result_file.write('%d\t%f   %f   %f   %f   %f\n' % (check_point['epoch'], mae, voxel_ious[0], voxel_ious[1], voxel_ious[2], voxel_ious[3]))
    # eval_result_file.close()

    print('Evaluate the model:', model_directory)

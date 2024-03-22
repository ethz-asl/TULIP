# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.


import math
import sys
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched
from torchvision.utils import make_grid

from pathlib import Path
import os
import numpy as np

# For Visualization
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from util.evaluation import *
from util.filter import *
import trimesh

import time
import tqdm

import json

cNorm = colors.Normalize(vmin=0, vmax=1)
jet = plt.get_cmap('viridis_r')
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

jet_loss_map = plt.get_cmap('jet')
scalarMap_loss_map = cmx.ScalarMappable(norm=cNorm, cmap=jet_loss_map)

def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    ema = None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples_low_res, samples_high_res) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        samples_low_res = samples_low_res['sample']
        samples_high_res = samples_high_res['sample']
        samples_low_res = samples_low_res.to(device, non_blocking=True)
        samples_high_res = samples_high_res.to(device, non_blocking=True)


        with torch.cuda.amp.autocast():
            _, total_loss, pixel_loss = model(samples_low_res, 
                            samples_high_res, 
                            eval = False)

        total_loss_value = total_loss.item()
        pixel_loss_value = pixel_loss.item()

        if not math.isfinite(total_loss_value):
            print("Total Loss is {}, stopping training".format(total_loss_value))
            print("Pixel Loss is {}, stopping training".format(pixel_loss_value))
            sys.exit(1)

        total_loss /= accum_iter
        loss_scaler(total_loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        
        if ema is not None:
            ema.update()

        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=total_loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        if args.log_transform or args.depth_scale_loss:
            total_loss_value_reduce = misc.all_reduce_mean(total_loss_value)
        pixel_loss_value_reduce = misc.all_reduce_mean(pixel_loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            if args.log_transform or args.depth_scale_loss:
                log_writer.add_scalar('train_loss_total', total_loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('train_loss_pixel', pixel_loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(data_loader, model, device, log_writer, args=None):
  
    '''Evaluation without Monte Carlo Dropout'''

    h_low_res = tuple(args.img_size_low_res)[0]
    h_high_res = tuple(args.img_size_high_res)[0]

    downsampling_factor = h_high_res // h_low_res

    # switch to evaluation mode
    model.eval()

    grid_size = args.grid_size
    global_step = 0
    total_loss = 0
    total_iou = 0
    total_cd = 0
    total_f1 = 0
    total_precision = 0
    total_recall = 0
    local_step = 0
    dataset_size = len(data_loader)

    evaluation_metrics = {'mae':[],
                          'chamfer_dist':[],
                          'iou':[],
                          'precision':[],
                          'recall':[],
                          'f1':[]}


    for batch in tqdm.tqdm(data_loader):

        images_low_res = batch[0]['sample'] # (B=1, C, H, W)
        images_high_res = batch[1]['sample'] # (B=1, C, H, W)

        images_low_res = images_low_res.to(device, non_blocking=True)
        images_high_res = images_high_res.to(device, non_blocking=True)
        
        global_step += 1
        # compute output
        with torch.cuda.amp.autocast():
            pred_img, _, _= model(images_low_res, 
                                    images_high_res, 
                                    eval = True)

            
        if log_writer is not None:

            # Preprocess the image
            if args.log_transform:
                pred_img = torch.expm1(pred_img)
                images_high_res = torch.expm1(images_high_res)
                images_low_res = torch.expm1(images_low_res)

            
            if args.dataset_select == "carla":
                pred_img = torch.where((pred_img >= 2/80) & (pred_img <= 1), pred_img, 0)
            elif args.dataset_select == "durlar":
                pred_img = torch.where((pred_img >= 0.3/120) & (pred_img <= 1), pred_img, 0)
            elif args.dataset_select == "kitti":
                pred_img = torch.where((pred_img >= 2/80) & (pred_img <= 1), pred_img, 0)
            else:
                print("Not Preprocess the pred image")
            
            loss_map = (pred_img -images_high_res).abs()
            pixel_loss_one_input = loss_map.mean()

            images_high_res = images_high_res.permute(0, 2, 3, 1).squeeze()
            images_low_res = images_low_res.permute(0, 2, 3, 1).squeeze()
            pred_img = pred_img.permute(0, 2, 3, 1).squeeze()
            

            images_high_res = images_high_res.detach().cpu().numpy()
            pred_img = pred_img.detach().cpu().numpy()
            images_low_res = images_low_res.detach().cpu().numpy()


            if args.dataset_select == "carla":

                if tuple(args.img_size_low_res)[1] != tuple(args.img_size_high_res)[1]:
                    loss_low_res_part = 0
                else:
                    low_res_index = range(0, h_high_res, downsampling_factor)
                    pred_low_res_part = pred_img[low_res_index, :]
                    loss_low_res_part = np.abs(pred_low_res_part - images_low_res)
                    loss_low_res_part = loss_low_res_part.mean()

                    pred_img[low_res_index, :] = images_low_res

                # pred_img = np.flip(pred_img)
                # images_high_res = np.flip(images_high_res)

                pcd_pred = img_to_pcd_carla(pred_img, maximum_range = 80)
                pcd_gt = img_to_pcd_carla(images_high_res, maximum_range = 80)
            
            elif args.dataset_select == "kitti":
                low_res_index = range(0, h_high_res, downsampling_factor)

                pred_low_res_part = pred_img[low_res_index, :]
                loss_low_res_part = np.abs(pred_low_res_part - images_low_res)
                loss_low_res_part = loss_low_res_part.mean()

                pred_img[low_res_index, :] = images_low_res

                # 3D Evaluation Metrics
                pcd_pred = img_to_pcd_kitti(pred_img, maximum_range= 80)
                pcd_gt = img_to_pcd_kitti(images_high_res, maximum_range = 80)


            elif args.dataset_select == "durlar":
                # Keep the pixel values in low resolution image
                low_res_index = range(0, h_high_res, downsampling_factor)

                # Evaluate the loss of low resolution part
                pred_low_res_part = pred_img[low_res_index, :]
                loss_low_res_part = np.abs(pred_low_res_part - images_low_res)
                loss_low_res_part = loss_low_res_part.mean()

                pred_img[low_res_index, :] = images_low_res

                if args.keep_close_scan:
                    pred_img[pred_img > 0.25] = 0
                    images_high_res[images_high_res > 0.25] = 0 

                # 3D Evaluation Metrics
                pcd_pred = img_to_pcd_durlar(pred_img, maximum_range= 120)
                pcd_gt = img_to_pcd_durlar(images_high_res, maximum_range = 120)
            else:
                raise NotImplementedError(f"Cannot find the dataset: {args.dataset_select}")


            pcd_all = np.vstack((pcd_pred, pcd_gt))

            chamfer_dist = chamfer_distance(pcd_gt, pcd_pred)


            min_coord = np.min(pcd_all, axis=0)
            max_coord = np.max(pcd_all, axis=0)
            

            # Voxelize the ground truth and prediction point clouds
            voxel_grid_predicted = voxelize_point_cloud(pcd_pred, grid_size, min_coord, max_coord)
            voxel_grid_ground_truth = voxelize_point_cloud(pcd_gt, grid_size, min_coord, max_coord)



            # Calculate metrics
            iou, precision, recall = calculate_metrics(voxel_grid_predicted, voxel_grid_ground_truth)
            f1 = 2 * (precision * recall) / (precision + recall)

            evaluation_metrics['mae'].append(pixel_loss_one_input.item())
            evaluation_metrics['chamfer_dist'].append(chamfer_dist.item())
            evaluation_metrics['iou'].append(iou)
            evaluation_metrics['precision'].append(precision)
            evaluation_metrics['recall'].append(recall)
            evaluation_metrics['f1'].append(f1)

            if global_step % 100 == 0 or global_step == 1:
                loss_map_normalized = (loss_map - loss_map.min()) / (loss_map.max() - loss_map.min() + 1e-8)

                loss_map_normalized = loss_map_normalized.permute(0, 2, 3, 1).squeeze()
                loss_map_normalized = loss_map_normalized.detach().cpu().numpy()
                loss_map_normalized = scalarMap_loss_map.to_rgba(loss_map_normalized)[..., :3]

                images_high_res = scalarMap.to_rgba(images_high_res)[..., :3]
                pred_img = scalarMap.to_rgba(pred_img)[..., :3]
                vis_grid = make_grid([torch.Tensor(images_high_res).permute(2, 0, 1), 
                                    torch.Tensor(pred_img).permute(2, 0, 1),
                                    torch.Tensor(loss_map_normalized).permute(2, 0, 1)], nrow=1)
                log_writer.add_image('gt - pred', vis_grid, local_step)
                log_writer.add_scalar('Test/mae_all', pixel_loss_one_input.item(), local_step)

                log_writer.add_scalar('Test/mae_low_res', loss_low_res_part, local_step)
                log_writer.add_scalar('Test/chamfer_dist', chamfer_dist, local_step)
                log_writer.add_scalar('Test/iou', iou, local_step)
                log_writer.add_scalar('Test/precision', precision, local_step)
                log_writer.add_scalar('Test/recall', recall, local_step)

                if args.save_pcd:
                
                    if local_step % 4 == 0:
                        pcd_outputpath = os.path.join(args.output_dir, 'pcd')
                        if not os.path.exists(pcd_outputpath):
                            os.mkdir(pcd_outputpath)
                        pcd_pred_color = np.zeros_like(pcd_pred)
                        pcd_pred_color[:, 0] = 255
                        pcd_gt_color = np.zeros_like(pcd_gt)
                        pcd_gt_color[:, 2] = 255
                        

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
            total_loss += pixel_loss_one_input.item()
            total_f1 += f1
            total_precision += precision
            total_recall += recall


    evaluation_file_path = os.path.join(args.output_dir,'results.txt')
    with open(evaluation_file_path, 'w') as file:
        json.dump(evaluation_metrics, file)

    print(print(f'Dictionary saved to {evaluation_file_path}'))

        
    # results = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    avg_loss = total_loss / global_step
    if log_writer is not None:
        log_writer.add_scalar('Metrics/test_average_iou', total_iou/global_step, 0)
        log_writer.add_scalar('Metrics/test_average_cd', total_cd/global_step, 0)
        log_writer.add_scalar('Metrics/test_average_loss', avg_loss, 0)
        log_writer.add_scalar('Metrics/test_average_f1', total_f1/global_step, 0)
        log_writer.add_scalar('Metrics/test_average_precision', total_precision/global_step, 0)
        log_writer.add_scalar('Metrics/test_average_recall', total_recall/global_step, 0)




# TODO: MC Drop
@torch.no_grad()
def MCdrop(data_loader, model, device, log_writer, args=None):
    '''Evaluation without Monte Carlo Dropout'''

    iteration = args.num_mcdropout_iterations
    iteration_batch = 8
    noise_threshold = args.noise_threshold

    assert iteration > iteration_batch 
    # metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'
    h_low_res = tuple(args.img_size_low_res)[0]
    h_high_res = tuple(args.img_size_high_res)[0]

    downsampling_factor = h_high_res // h_low_res

    # keep model in train mode to enable Dropout
    model.eval()
    enable_dropout(model)

    grid_size = args.grid_size
    global_step = 0
    total_loss = 0
    local_step = 0
    total_iou = 0
    total_cd = 0
    total_f1 = 0
    total_precision = 0
    total_recall = 0

    evaluation_metrics = {'mae':[],
                          'chamfer_dist':[],
                          'iou':[],
                          'precision':[],
                          'recall':[],
                          'f1':[]}

    for batch in tqdm.tqdm(data_loader):


        images_low_res = batch[0]['sample'] # (B=1, C, H, W)
        images_high_res = batch[1]['sample'] # (B=1, C, H, W)

        images_low_res = images_low_res.to(device, non_blocking=True)
        images_high_res = images_high_res.to(device, non_blocking=True)
        global_step += 1
        # compute output

        with torch.cuda.amp.autocast():
            
            pred_img_iteration = torch.empty(iteration, images_high_res.shape[1], images_high_res.shape[2], images_high_res.shape[3]).to(device)
            for i in range(int(np.ceil(iteration / iteration_batch))):
                input_batch = iteration_batch if (iteration-i*iteration_batch) > iteration_batch else (iteration-i*iteration_batch)
                test_imgs_input = torch.tile(images_low_res, (input_batch, 1, 1, 1))
                

                pred_imgs = model(test_imgs_input, 
                                images_high_res, 
                                mc_drop = True) 
                
                pred_img_iteration[i*iteration_batch:i*iteration_batch+input_batch, ...] = pred_imgs
            pred_img = torch.mean(pred_img_iteration, dim = 0, keepdim = True)
            pred_img_var = torch.std(pred_img_iteration, dim = 0, keepdim = True)
            noise_removal = pred_img_var > noise_threshold * pred_img
            
            pred_img[noise_removal] = 0

        if log_writer is not None:

            if args.log_transform:
                pred_img = torch.expm1(pred_img)
                images_high_res = torch.expm1(images_high_res)
                images_low_res = torch.expm1(images_low_res)
            

             # Preprocess the image
            if args.dataset_select == "carla":
                pred_img = torch.where((pred_img >= 2/80) & (pred_img <= 1), pred_img, 0)
            elif args.dataset_select == "durlar":
                pred_img = torch.where((pred_img >= 0.3/120) & (pred_img <= 1), pred_img, 0)
            elif args.dataset_select == "kitti":
                pred_img = torch.where((pred_img >= 0) & (pred_img <= 1), pred_img, 0)
            else:
                print("Not Preprocess the pred image")
            
            loss_map = (pred_img -images_high_res).abs()
            pixel_loss_one_input = loss_map.mean()
        
            
            images_high_res = images_high_res.permute(0, 2, 3, 1).squeeze()
            images_low_res = images_low_res.permute(0, 2, 3, 1).squeeze()
            pred_img = pred_img.permute(0, 2, 3, 1).squeeze()

            

            images_high_res = images_high_res.detach().cpu().numpy()
            pred_img = pred_img.detach().cpu().numpy()
            images_low_res = images_low_res.detach().cpu().numpy()

            if args.dataset_select == "carla":
                if tuple(args.img_size_low_res)[1] != tuple(args.img_size_high_res)[1]:
                    loss_low_res_part = 0
                else:
                    low_res_index = range(0, h_high_res, downsampling_factor)

                    # Evaluate the loss of low resolution part
                    pred_low_res_part = pred_img[low_res_index, :]
                    loss_low_res_part = np.abs(pred_low_res_part - images_low_res)
                    loss_low_res_part = loss_low_res_part.mean()

                    pred_img[low_res_index, :] = images_low_res

                # pred_img = np.flip(pred_img)
                # images_high_res = np.flip(images_high_res)

                pcd_pred = img_to_pcd_carla(pred_img, maximum_range = 80)
                pcd_gt = img_to_pcd_carla(images_high_res, maximum_range = 80)

            elif args.dataset_select == "kitti":
                low_res_index = range(0, h_high_res, downsampling_factor)

                # Evaluate the loss of low resolution part
                pred_low_res_part = pred_img[low_res_index, :]
                loss_low_res_part = np.abs(pred_low_res_part - images_low_res)
                loss_low_res_part = loss_low_res_part.mean()

                pred_img[low_res_index, :] = images_low_res

                if args.keep_close_scan:
                    pred_img[pred_img > 0.25] = 0
                    images_high_res[images_high_res > 0.25] = 0 

                # 3D Evaluation Metrics
                pcd_pred = img_to_pcd_kitti(pred_img, maximum_range= 80)
                pcd_gt = img_to_pcd_kitti(images_high_res, maximum_range = 80)

            elif args.dataset_select == "durlar":
                # Keep the pixel values in low resolution image
                low_res_index = range(0, h_high_res, downsampling_factor)

                # Evaluate the loss of low resolution part
                pred_low_res_part = pred_img[low_res_index, :]
                loss_low_res_part = np.abs(pred_low_res_part - images_low_res)
                loss_low_res_part = loss_low_res_part.mean()

                pred_img[low_res_index, :] = images_low_res
                

                pcd_pred = img_to_pcd_durlar(pred_img)
                pcd_gt = img_to_pcd_durlar(images_high_res)
            
            else:
                raise NotImplementedError(f"Cannot find the dataset: {args.dataset_select}")

            pcd_all = np.vstack((pcd_pred, pcd_gt))

            chamfer_dist = chamfer_distance(pcd_gt, pcd_pred)
            min_coord = np.min(pcd_all, axis=0)
            max_coord = np.max(pcd_all, axis=0)
            
            # Voxelize the ground truth and prediction point clouds
            voxel_grid_predicted = voxelize_point_cloud(pcd_pred, grid_size, min_coord, max_coord)
            voxel_grid_ground_truth = voxelize_point_cloud(pcd_gt, grid_size, min_coord, max_coord)
            # Calculate metrics
            iou, precision, recall = calculate_metrics(voxel_grid_predicted, voxel_grid_ground_truth)

            f1 = 2 * (precision * recall) / (precision + recall)

            evaluation_metrics['mae'].append(pixel_loss_one_input.item())
            evaluation_metrics['chamfer_dist'].append(chamfer_dist.item())
            evaluation_metrics['iou'].append(iou)
            evaluation_metrics['precision'].append(precision)
            evaluation_metrics['recall'].append(recall)
            evaluation_metrics['f1'].append(f1)
            
            if global_step % 100 == 0 or global_step == 1:
                loss_map_normalized = (loss_map - loss_map.min()) / (loss_map.max() - loss_map.min() + 1e-8)
                loss_map_normalized = loss_map_normalized.permute(0, 2, 3, 1).squeeze()
                loss_map_normalized = loss_map_normalized.detach().cpu().numpy()
                loss_map_normalized = scalarMap_loss_map.to_rgba(loss_map_normalized)[..., :3]

                images_high_res = scalarMap.to_rgba(images_high_res)[..., :3]
                pred_img = scalarMap.to_rgba(pred_img)[..., :3]
                vis_grid = make_grid([torch.Tensor(images_high_res).permute(2, 0, 1), 
                                    torch.Tensor(pred_img).permute(2, 0, 1),
                                    torch.Tensor(loss_map_normalized).permute(2, 0, 1)], nrow=1)

                log_writer.add_image('gt - pred', vis_grid, local_step)
                log_writer.add_scalar('Test/mae_all', pixel_loss_one_input.item(), local_step)

                log_writer.add_scalar('Test/mae_low_res', loss_low_res_part, local_step)
                log_writer.add_scalar('Test/chamfer_dist', chamfer_dist, local_step)
                log_writer.add_scalar('Test/iou', iou, local_step)
                log_writer.add_scalar('Test/precision', precision, local_step)
                log_writer.add_scalar('Test/recall', recall, local_step)

                if args.save_pcd:
                    
                    if local_step % 4 == 0:
                        # pcd_outputpath = os.path.join(args.output_dir, 'pcd_mc_drop_smaller_noise_threshold')
                        pcd_outputpath = os.path.join(args.output_dir, 'pcd_mc_drop')
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

                # exit(0)

                

                local_step += 1

            total_iou += iou
            total_cd += chamfer_dist
            total_loss += pixel_loss_one_input.item()
            total_f1 += f1
            total_precision += precision
            total_recall += recall

    evaluation_file_path = os.path.join(args.output_dir,'results_mcdrop.txt')
    with open(evaluation_file_path, 'w') as file:
        json.dump(evaluation_metrics, file) 

    print(print(f'Dictionary saved to {evaluation_file_path}'))

    avg_loss = total_loss / global_step
    if log_writer is not None:
        log_writer.add_scalar('Metrics/test_average_iou', total_iou/global_step, 0)
        log_writer.add_scalar('Metrics/test_average_cd', total_cd/global_step, 0)
        log_writer.add_scalar('Metrics/test_average_loss', avg_loss, 0)
        log_writer.add_scalar('Metrics/test_average_f1', total_f1/global_step, 0)
        log_writer.add_scalar('Metrics/test_average_precision', total_precision/global_step, 0)
        log_writer.add_scalar('Metrics/test_average_recall', total_recall/global_step, 0)


def get_latest_checkpoint(args):
    output_dir = Path(args.output_dir)
    import glob
    all_checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint-*.pth'))
    latest_ckpt = -1
    for ckpt in all_checkpoints:
        t = ckpt.split('-')[-1].split('.')[0]
        if t.isdigit():
            latest_ckpt = max(int(t), latest_ckpt)
    if latest_ckpt >= 0:
        args.resume = os.path.join(output_dir, 'checkpoint-%d.pth' % latest_ckpt)
    print("Find checkpoint: %s" % args.resume)

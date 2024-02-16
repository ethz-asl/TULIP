# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import sys
from typing import Iterable

import torch
from timm.utils import accuracy

import util.misc as misc
import util.lr_sched as lr_sched
from torchvision.utils import make_grid
import trimesh
from util.evaluation import *
from util.filter import *

from pathlib import Path
import os
import numpy as np

# For Visualization
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx

import tqdm

cNorm = colors.Normalize(vmin=0, vmax=1)
jet = plt.get_cmap('viridis_r')
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

jet_loss_map = plt.get_cmap('jet')
scalarMap_loss_map = cmx.ScalarMappable(norm=cNorm, cmap=jet_loss_map)


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
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

    for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        #  Inverse, closer object has higher pixel value, but keep black for pixels without LiDAR Return
        if args.reverse_pixel_value:
            samples = 1-samples
            samples[samples == 1] = 0
    
        samples = samples.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            # Only Swin MAE is able to train with curriculum learning strategy, as it uses all patchs (mask and non-mask) as input
            if args.curriculum_learning and args.model_select == "swin_mae":
                curriculum_step = args.epochs // 10 # 1 + 2 + 3
                if epoch < 7*curriculum_step:
                    args.mask_ratio = 0.75
                elif epoch >= 7*curriculum_step and epoch < 9*curriculum_step:
                    args.mask_ratio = 0.5
                else:
                    args.mask_ratio = 0.25
            loss, _, _ = model(samples, mask_ratio=args.mask_ratio, mask_loss = args.mask_loss, loss_on_unmasked = args.loss_on_unmasked)
            

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(data_loader, model, device, log_writer, args=None):
    # This criterion is also for classfiction, we can directly use the loss forward computation in mae model
    # criterion = torch.nn.CrossEntropyLoss()

    # metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    global_step = 0
    total_loss = 0
    total_iou = 0
    total_cd = 0
    local_step = 0
    grid_size = 0.1
    is_low_res = (tuple(args.img_size)[0] < 128)
    # iterator = iter(data_loader)

    # test_data = []
    for batch in tqdm.tqdm(data_loader):
        images = batch[0] # (B=1, C, H, W)

        #  Inverse, closer object has higher pixel value, but keep black for pixels without LiDAR Return
        if args.reverse_pixel_value:
            images = 1-images
            images[images == 1] = 0

        # target = batch[-1]
        images = images.to(device, non_blocking=True)
        # target = target.to(device, non_blocking=True)
        global_step += 1
        # compute output
        with torch.cuda.amp.autocast():
            if args.curriculum_learning and args.model_select == "swin_mae":
                args.mask_ratio = 0.75
            # args.mask_ratio = 0.75
            output = model(images, mask_ratio = args.mask_ratio, mask_loss = args.mask_loss, eval = True, loss_on_unmasked = args.loss_on_unmasked) # --> tuple(loss, pred_imgs, masked_imgs)
            loss = output[0]
            pred_img = output[1] # (B=1, C, H, W)
            masked_img = output[2] # (B=1, C, H, W) B: Batch Size, N_MASK: Number of masks H*W/(patch_size*patch_size)

            total_loss += loss.item()
       
            # loss = criterion(output[1], target)
        if log_writer is not None:

            if args.in_chans == 4:
                # pred_img = depth_wise_unconcate(pred_img)
                images = depth_wise_unconcate(images)
                # masked_img = depth_wise_unconcate(masked_img)
                # loss_map = depth_wise_unconcate(loss_map)

            if args.log_transform:
                # print("Logarithmus Transform back the prediction and ground-truth images")
                total_loss_one_input = (pred_img - images).abs().mean()
                pred_img = torch.expm1(pred_img)
                images = torch.expm1(images)
            
            


            if args.dataset_select == "carla":
                pred_img = torch.where((pred_img >= 2/80) & (pred_img <= 1), pred_img, 0)
            elif args.dataset_select == "durlar":
                pred_img = torch.where((pred_img >= 0.3/120) & (pred_img <= 1), pred_img, 0)
            else:
                print("Not Preprocess the pred image")


            loss_map = (pred_img -images).abs()
            pixel_loss_one_input = loss_map.mean()

            images = images.permute(0, 2, 3, 1).squeeze()
            pred_img = pred_img.permute(0, 2, 3, 1).squeeze()
            # masked_img = masked_img.permute(0, 2, 3, 1).squeeze()
            images = images.detach().cpu().numpy()
            pred_img = pred_img.detach().cpu().numpy()
            # masked_img = masked_img.detach().cpu().numpy()
            loss_map = loss_map.permute(0, 2, 3, 1).squeeze()
            loss_map = loss_map.detach().cpu().numpy()

            


            if args.dataset_select == "carla":

                pred_img = np.flip(pred_img)
                images = np.flip(images)

                pcd_pred = img_to_pcd_carla(pred_img, maximum_range = 80)
                pcd_gt = img_to_pcd_carla(images, maximum_range = 80)

            else:
                # 3D Evaluation Metrics
                pcd_pred = img_to_pcd(pred_img, maximum_range= 120)
                pcd_gt = img_to_pcd(images, maximum_range = 120)


            pcd_all = np.vstack((pcd_pred, pcd_gt))

            chamfer_dist = chamfer_distance(pcd_gt, pcd_pred)

            # time_cd = time.time() - time_proj
            # print("Time for chamfer distance", time_cd)

            min_coord = np.min(pcd_all, axis=0)
            max_coord = np.max(pcd_all, axis=0)
            

            # Voxelize the ground truth and prediction point clouds
            voxel_grid_predicted = voxelize_point_cloud(pcd_pred, grid_size, min_coord, max_coord)
            voxel_grid_ground_truth = voxelize_point_cloud(pcd_gt, grid_size, min_coord, max_coord)


            # time_voxelize = time.time() - time_cd
            # print("Time for voxelize", time_voxelize)


            # Calculate metrics
            iou, precision, recall = calculate_metrics(voxel_grid_predicted, voxel_grid_ground_truth)

            if global_step % 100 == 0 or global_step == 1:
                loss_map_normalized = (loss_map - loss_map.min()) / (loss_map.max() - loss_map.min() + 1e-8)
                loss_map_normalized = scalarMap_loss_map.to_rgba(loss_map_normalized)[..., :3]

                images = scalarMap.to_rgba(images)[..., :3]
                pred_img = scalarMap.to_rgba(pred_img)[..., :3]

                # mask_for_masked_image = masked_img == 1
                # mask_for_masked_image = mask_for_masked_image.squeeze() 
                # masked_img = scalarMap.to_rgba(masked_img)[..., :3]
                # masked_img[mask_for_masked_image,:] = [1, 1, 1]
                # vis_grid = make_grid(torch.cat([images, pred_img, masked_img], dim = 0), nrow=1)
                vis_grid = make_grid([torch.Tensor(images).permute(2, 0, 1), 
                                    torch.Tensor(pred_img).permute(2, 0, 1),
                                    # torch.Tensor(masked_img).permute(2, 0, 1),
                                    torch.Tensor(loss_map_normalized).permute(2, 0, 1)], nrow=1)
                if args.log_transform or args.depth_scale_loss:
                    log_writer.add_scalar('Test/logtransform_mse_all', total_loss_one_input.item(), local_step)
                log_writer.add_image('gt - pred - mask', vis_grid, local_step)
                log_writer.add_scalar('Test/mse_all', pixel_loss_one_input.item(), local_step)

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
                        
                        # pcd_all_color = np.vstack((pcd_pred_color, pcd_gt_color))

                        point_cloud_pred = trimesh.PointCloud(
                            vertices=pcd_pred,
                            colors=pcd_pred_color)
                        
                        point_cloud_gt = trimesh.PointCloud(
                            vertices=pcd_gt,
                            colors=pcd_gt_color)
                        
                        point_cloud_pred.export(os.path.join(pcd_outputpath, f"pred_{local_step}.ply"))  
                        point_cloud_gt.export(os.path.join(pcd_outputpath, f"gt_{local_step}.ply")) 

                local_step += 1

            total_iou += iou
            total_cd += chamfer_dist
            total_loss += pixel_loss_one_input.item()
        
    
    avg_loss = total_loss / global_step
    if log_writer is not None:
        log_writer.add_scalar('Metrics/test_average_iou', total_iou/global_step, 0)
        log_writer.add_scalar('Metrics/test_average_cd', total_cd/global_step, 0)
        log_writer.add_scalar('Metrics/test_average_loss', avg_loss, 0)

@torch.no_grad()
def MCdrop(data_loader, model, device, log_writer, args=None):
    # This criterion is also for classfiction, we can directly use the loss forward computation in mae model
    # criterion = torch.nn.CrossEntropyLoss()

    # metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    iteration = 50
    iteration_batch = 8
    noise_threshold = args.noise_threshold

    global_step = 0
    total_loss = 0
    total_iou = 0
    total_cd = 0
    local_step = 0
    grid_size = 0.1
    is_low_res = (tuple(args.img_size)[0] < 128)
    # iterator = iter(data_loader)

    # test_data = []
    for batch in tqdm.tqdm(data_loader):
        images = batch[0] # (B=1, C, H, W)

        #  Inverse, closer object has higher pixel value, but keep black for pixels without LiDAR Return
        if args.reverse_pixel_value:
            images = 1-images
            images[images == 1] = 0

        # target = batch[-1]
        images = images.to(device, non_blocking=True)
        # target = target.to(device, non_blocking=True)
        global_step += 1
        # compute output
        with torch.cuda.amp.autocast():
            if args.curriculum_learning and args.model_select == "swin_mae":
                args.mask_ratio = 0.75
            # args.mask_ratio = 0.75

            pred_img_iteration = torch.empty(iteration, images.shape[1], images.shape[2], images.shape[3]).to(device)
            for i in range(int(np.ceil(iteration / iteration_batch))):
                input_batch = iteration_batch if (iteration-i*iteration_batch) > iteration_batch else (iteration-i*iteration_batch)
                test_imgs_input = torch.tile(images, (input_batch, 1, 1, 1))
                
                output = model(test_imgs_input, mask_ratio = args.mask_ratio, mask_loss = args.mask_loss, eval = True, loss_on_unmasked = args.loss_on_unmasked)
                pred_imgs = output[1]

                pred_img_iteration[i*iteration_batch:i*iteration_batch+input_batch, ...] = pred_imgs
            pred_img = torch.mean(pred_img_iteration, dim = 0, keepdim = True)
            pred_img_var = torch.std(pred_img_iteration, dim = 0, keepdim = True)
            noise_removal = pred_img_var > noise_threshold * pred_img
            
            pred_img[noise_removal] = 0
            loss = output[0]
            masked_img = output[2][:1] # (B=1, C, H, W) B: Batch Size, N_MASK: Number of masks H*W/(patch_size*patch_size)

            total_loss += loss.item()

            # loss = criterion(output[1], target)
        if log_writer is not None:

            if args.log_transform:
                # print("Logarithmus Transform back the prediction and ground-truth images")
                total_loss_one_input = (pred_img - images).abs().mean()
                pred_img = torch.expm1(pred_img)
                images = torch.expm1(images)
            
            


            if args.dataset_select == "carla":
                pred_img = torch.where((pred_img >= 2/80) & (pred_img <= 1), pred_img, 0)
            elif args.dataset_select == "durlar":
                pred_img = torch.where((pred_img >= 0.3/120) & (pred_img <= 1), pred_img, 0)
            else:
                print("Not Preprocess the pred image")


            loss_map = (pred_img -images).abs()
            pixel_loss_one_input = loss_map.mean()

            images = images.permute(0, 2, 3, 1).squeeze()
            pred_img = pred_img.permute(0, 2, 3, 1).squeeze()
            masked_img = masked_img.permute(0, 2, 3, 1).squeeze()
            images = images.detach().cpu().numpy()
            pred_img = pred_img.detach().cpu().numpy()
            masked_img = masked_img.detach().cpu().numpy()
            loss_map = loss_map.permute(0, 2, 3, 1).squeeze()
            loss_map = loss_map.detach().cpu().numpy()

            if args.in_chans == 4:
                pred_img = depth_wise_unconcate(pred_img)
                images = depth_wise_unconcate(images)
                masked_img = depth_wise_unconcate(masked_img)
                loss_map = depth_wise_unconcate(loss_map)


            if args.dataset_select == "carla":

                pred_img = np.flip(pred_img)
                images = np.flip(images)

                pcd_pred = img_to_pcd_carla(pred_img, maximum_range = 80)
                pcd_gt = img_to_pcd_carla(images, maximum_range = 80)

            else:
                # 3D Evaluation Metrics
                pcd_pred = img_to_pcd(pred_img, maximum_range= 120)
                pcd_gt = img_to_pcd(images, maximum_range = 120)


            pcd_all = np.vstack((pcd_pred, pcd_gt))

            chamfer_dist = chamfer_distance(pcd_gt, pcd_pred)

            # time_cd = time.time() - time_proj
            # print("Time for chamfer distance", time_cd)

            min_coord = np.min(pcd_all, axis=0)
            max_coord = np.max(pcd_all, axis=0)
            

            # Voxelize the ground truth and prediction point clouds
            voxel_grid_predicted = voxelize_point_cloud(pcd_pred, grid_size, min_coord, max_coord)
            voxel_grid_ground_truth = voxelize_point_cloud(pcd_gt, grid_size, min_coord, max_coord)


            # time_voxelize = time.time() - time_cd
            # print("Time for voxelize", time_voxelize)


            # Calculate metrics
            iou, precision, recall = calculate_metrics(voxel_grid_predicted, voxel_grid_ground_truth)

            if global_step % 100 == 0 or global_step == 1:
                loss_map_normalized = (loss_map - loss_map.min()) / (loss_map.max() - loss_map.min() + 1e-8)
                loss_map_normalized = scalarMap_loss_map.to_rgba(loss_map_normalized)[..., :3]

                images = scalarMap.to_rgba(images)[..., :3]
                pred_img = scalarMap.to_rgba(pred_img)[..., :3]

                mask_for_masked_image = masked_img == 1
                mask_for_masked_image = mask_for_masked_image.squeeze() 
                masked_img = scalarMap.to_rgba(masked_img)[..., :3]
                masked_img[mask_for_masked_image,:] = [1, 1, 1]
                # vis_grid = make_grid(torch.cat([images, pred_img, masked_img], dim = 0), nrow=1)
                vis_grid = make_grid([torch.Tensor(images).permute(2, 0, 1), 
                                    torch.Tensor(pred_img).permute(2, 0, 1),
                                    torch.Tensor(masked_img).permute(2, 0, 1),
                                    torch.Tensor(loss_map_normalized).permute(2, 0, 1)], nrow=1)
                if args.log_transform or args.depth_scale_loss:
                    log_writer.add_scalar('Test/logtransform_mse_all', total_loss_one_input.item(), local_step)
                log_writer.add_image('gt - pred - mask', vis_grid, local_step)
                log_writer.add_scalar('Test/mse_all', pixel_loss_one_input.item(), local_step)

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
                        
                        # pcd_all_color = np.vstack((pcd_pred_color, pcd_gt_color))

                        point_cloud_pred = trimesh.PointCloud(
                            vertices=pcd_pred,
                            colors=pcd_pred_color)
                        
                        point_cloud_gt = trimesh.PointCloud(
                            vertices=pcd_gt,
                            colors=pcd_gt_color)
                        
                        point_cloud_pred.export(os.path.join(pcd_outputpath, f"pred_{local_step}.ply"))  
                        point_cloud_gt.export(os.path.join(pcd_outputpath, f"gt_{local_step}.ply")) 

                local_step += 1

            total_iou += iou
            total_cd += chamfer_dist
            total_loss += pixel_loss_one_input.item()
        
    
    avg_loss = total_loss / global_step
    if log_writer is not None:
        log_writer.add_scalar('Metrics/test_average_iou', total_iou/global_step, 0)
        log_writer.add_scalar('Metrics/test_average_cd', total_cd/global_step, 0)
        log_writer.add_scalar('Metrics/test_average_loss', avg_loss, 0)

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

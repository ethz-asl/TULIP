# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable

import torch
from timm.utils import accuracy

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

cNorm = colors.Normalize(vmin=0, vmax=1)
jet = plt.get_cmap('viridis_r')
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable,
                    optimizer: torch.optim.Optimizer,
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

    for data_iter_step, (samples_low_res, samples_high_res) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        samples_low_res = samples_low_res[0]
        samples_high_res = samples_high_res[0]
        samples_low_res = samples_low_res.to(device, non_blocking=True)
        samples_high_res = samples_high_res.to(device, non_blocking=True)


        with torch.cuda.amp.autocast():
            _, total_loss, pixel_loss = model(samples_low_res.contiguous(), 
                            samples_high_res.contiguous(), 
                            img_size_high_res = args.img_size_high_res,
                            eval = False)

        # loss = criterion(pred, samples_high_res.contiguous())
        total_loss_value = total_loss.item()
        pixel_loss_value = pixel_loss.item()

        if not math.isfinite(total_loss_value):
            print("Total Loss is {}, stopping training".format(total_loss_value))
            print("Pixel Loss is {}, stopping training".format(pixel_loss_value))
            sys.exit(1)

        total_loss /= accum_iter
        loss_scaler(total_loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=total_loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        # total_loss_value_reduce = misc.all_reduce_mean(total_loss_value)
        pixel_loss_value_reduce = misc.all_reduce_mean(pixel_loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            # log_writer.add_scalar('train_loss_total', total_loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('train_loss_pixel', pixel_loss_value_reduce, epoch_1000x)
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
    h_low_res = tuple(args.img_size_low_res)[0]
    h_high_res = tuple(args.img_size_high_res)[0]

    downsampling_factor = h_high_res // h_low_res

    # switch to evaluation mode
    model.eval()

    grid_size = 0.1
    global_step = 0
    total_loss = 0
    local_step = 0
    # iterator = iter(data_loader)
    for batch in data_loader:
        images_low_res = batch[0][0] # (B=1, C, H, W)
        images_high_res = batch[1][0] # (B=1, C, H, W)

        # target = batch[-1]
        images_low_res = images_low_res.to(device, non_blocking=True)
        images_high_res = images_high_res.to(device, non_blocking=True)
        # target = target.to(device, non_blocking=True)
        global_step += 1
        # compute output
        with torch.cuda.amp.autocast():
            pred_img, total_loss_one_input, pixel_loss_one_input, horizontal_cat, vertical_cat = model(images_low_res, 
                                                                 images_high_res, 
                                                                 img_size_high_res = args.img_size_high_res,
                                                                 eval = True) # --> tuple(loss, pred_imgs, masked_imgs)
        # loss = criterion(pred_img, images_high_res)

            
        if log_writer is not None:
            images_high_res = images_high_res.permute(0, 2, 3, 1).squeeze()
            images_low_res = images_low_res.permute(0, 2, 3, 1).squeeze()
            pred_img = pred_img.permute(0, 2, 3, 1).squeeze()

            

            images_high_res = images_high_res.detach().cpu().numpy()
            pred_img = pred_img.detach().cpu().numpy()
            images_low_res = images_low_res.detach().cpu().numpy()
            # print(images_high_res.max())
            # exit(0)

            if args.log_transform:
                images_high_res = np.expm1(images_high_res)
                pred_img = np.expm1(pred_img)
                images_low_res = np.expm1(images_low_res)


            # # Test Logarithm Transformation

            # all_pixels = images_high_res.reshape(-1)
            # log_pixels = np.log1p(all_pixels + 1e-8)
            # # Creating the histogram
            # plt.hist(all_pixels, bins=30, edgecolor='black')
            # plt.hist(log_pixels, bins=30, edgecolor='red')
            # plt.savefig('data_distribution.png')

            # exit(0)

            # Keep the pixel values in low resolution image
            low_res_index = range(0, h_high_res, downsampling_factor)

            # Evaluate the loss of low resolution part
            pred_low_res_part = pred_img[low_res_index, :]
            loss_low_res_part = (pred_low_res_part - images_low_res) ** 2
            loss_low_res_part = loss_low_res_part.mean()

            pred_img[low_res_index, :] = images_low_res

            # # Test with edge detector
            # pred_edges = cv2.Canny((pred_img*255).astype(np.uint8), threshold1=30, threshold2=100)
            # gt_edges = cv2.Canny((images_high_res*255).astype(np.uint8), threshold1=30, threshold2=100)

            # 3D Evaluation Metrics
            pcd_pred = img_to_pcd(pred_img)
            pcd_gt = img_to_pcd(images_high_res)

            pcd_all = np.vstack((pcd_pred, pcd_gt))

            chamfer_dist = chamfer_distance(pcd_gt, pcd_pred)
            min_coord = np.min(pcd_all, axis=0)
            max_coord = np.max(pcd_all, axis=0)
            
            # Voxelize the ground truth and prediction point clouds
            voxel_grid_predicted = voxelize_point_cloud(pcd_pred, grid_size, min_coord, max_coord)
            voxel_grid_ground_truth = voxelize_point_cloud(pcd_gt, grid_size, min_coord, max_coord)
            # Calculate metrics
            iou, precision, recall = calculate_metrics(voxel_grid_predicted, voxel_grid_ground_truth)

            if args.save_pcd:
                
                pcd_outputpath = os.path.join(args.output_dir, 'pcd')
                if not os.path.exists(pcd_outputpath):
                    os.mkdir(pcd_outputpath)
                pcd_pred_color = np.zeros_like(pcd_pred)
                pcd_pred_color[:, 0] = 255
                pcd_gt_color = np.zeros_like(pcd_gt)
                pcd_gt_color[:, 2] = 255
                
                pcd_all_color = np.vstack((pcd_pred_color, pcd_gt_color))

                point_cloud = trimesh.PointCloud(
                    vertices=pcd_all,
                    colors=pcd_all_color)
                
                point_cloud.export(os.path.join(pcd_outputpath, f"pred_gt_{local_step}.ply"))     

            # exit(0)

            images_high_res = scalarMap.to_rgba(images_high_res)[..., :3]
            pred_img = scalarMap.to_rgba(pred_img)[..., :3]
            # vis_grid = make_grid(torch.cat([images, pred_img, masked_img], dim = 0), nrow=1)
            vis_grid = make_grid([torch.Tensor(images_high_res).permute(2, 0, 1), torch.Tensor(pred_img).permute(2, 0, 1)], nrow=1)
            if args.edge_loss:
                vis_grid_horizontal = make_grid(horizontal_cat, nrow=1)
                # vis_grid_vertical= make_grid(vertical_cat, nrow=1)
                log_writer.add_image('gt - pred (horizontal)', vis_grid_horizontal, local_step)
                # log_writer.add_image('gt - pred (vertical)', vis_grid_vertical, local_step)
                log_writer.add_scalar('Test/total_mse_all', total_loss_one_input.item(), local_step)
            # vis_grid_egde = make_grid([torch.Tensor(np.expand_dims(gt_edges / 255, axis = -1)).permute(2, 0, 1), torch.Tensor(np.expand_dims(pred_edges / 255, axis = -1)).permute(2, 0, 1)], nrow=1)
            log_writer.add_image('gt - pred', vis_grid, local_step)
            # log_writer.add_image('gt - pred (edges)', vis_grid_egde, local_step)
            log_writer.add_scalar('Test/mse_all', pixel_loss_one_input.item(), local_step)

            log_writer.add_scalar('Test/mse_low_res', loss_low_res_part, local_step)
            log_writer.add_scalar('Test/chamfer_dist', chamfer_dist, local_step)
            log_writer.add_scalar('Test/iou', iou, local_step)
            log_writer.add_scalar('Test/precision', precision, local_step)
            log_writer.add_scalar('Test/recall', recall, local_step)

            local_step += 1
        
        
        total_loss += pixel_loss_one_input.item()
    
    # results = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    avg_loss = total_loss / global_step
    if log_writer is not None:
        log_writer.add_scalar('Loss/test_average_loss', avg_loss, 0)



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

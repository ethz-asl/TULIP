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
from util.datasets import CropRanges 
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

@torch.no_grad()
def evaluate_diff_ranges(data_loader, model, device, log_writer, args=None):
    # This criterion is also for classfiction, we can directly use the loss forward computation in mae model
    # criterion = torch.nn.CrossEntropyLoss()

    # metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'
    h_low_res = tuple(args.img_size_low_res)[0]
    h_high_res = tuple(args.img_size_high_res)[0]

    downsampling_factor = h_high_res // h_low_res

    # switch to evaluation mode
    model.eval()

    grid_size = args.grid_size

    # 10m as range

    if args.dataset_select in ["carla", "carla200000"]:
        iteration_crop_ranges = 8
        max_range = 80
    elif args.dataset_select == "durlar":
        iteration_crop_ranges = 12
        max_range = 120
    elif args.dataset_select == "kitti":
        iteration_crop_ranges = 8
        max_range = 120
    else:
        print("Not Preprocess the pred image")

    evaluation_metrics = {'mae':[],
                          'chamfer_dist':[],
                          'iou':[],
                          'precision':[],
                          'recall':[],
                          'f1':[]}

    # iterator = iter(data_loader)
    

    global_step = 0
    total_loss = np.zeros(iteration_crop_ranges)
    total_iou = np.zeros(iteration_crop_ranges)
    total_cd = np.zeros(iteration_crop_ranges)
    total_f1 = np.zeros(iteration_crop_ranges)
    total_precision = np.zeros(iteration_crop_ranges)
    total_recall = np.zeros(iteration_crop_ranges)

    for batch in tqdm.tqdm(data_loader):

            # start_time = time.time()


        images_low_res = batch[0][0] # (B=1, C, H, W)
        images_high_res = batch[1][0] # (B=1, C, H, W)
        
        
        # target = batch[-1]
        images_low_res = images_low_res.to(device, non_blocking=True)
        images_high_res = images_high_res.to(device, non_blocking=True)
            # target = target.to(device, non_blocking=True)
        
        global_step += 1
        # compute output
        with torch.cuda.amp.autocast():
            pred_img, _, _= model(images_low_res, 
                                    images_high_res, 
                                    img_size_high_res = args.img_size_high_res,
                                    eval = True) # --> tuple(loss, pred_imgs, masked_imgs)
                
            # loss = criterion(pred_img, images_high_res)
        pred_img_full = pred_img.clone()
        images_high_res_full = images_high_res.clone()
        images_low_res_full = images_low_res.clone()
        for i in range(iteration_crop_ranges):
            min_dist = i*10 / max_range
            max_dist = (i+1)*10 / max_range
            range_cropper = CropRanges(min_dist, max_dist)

            pred_img,_= range_cropper(pred_img_full)
            images_high_res, num_pixels = range_cropper(images_high_res_full)
            images_low_res,_ = range_cropper(images_low_res_full)
        

            
            if log_writer is not None:
                # Visulize less for carla dataset

                # Preprocess the image
                if args.log_transform:
                    # print("Logarithmus Transform back the prediction and ground-truth images")
                    # total_loss_one_input = (pred_img - images_high_res).abs().mean()
                    pred_img = torch.expm1(pred_img)
                    images_high_res = torch.expm1(images_high_res)
                    images_low_res = torch.expm1(images_low_res)

                
                if args.dataset_select in ["carla", "carla200000"]:
                    pred_img = torch.where((pred_img >= 2/80) & (pred_img <= 1), pred_img, 0)
                elif args.dataset_select == "durlar":
                    pred_img = torch.where((pred_img >= 0.3/120) & (pred_img <= 1), pred_img, 0)
                elif args.dataset_select == "kitti":
                    pred_img = torch.where((pred_img >= 0) & (pred_img <= 1), pred_img, 0)
                else:
                    print("Not Preprocess the pred image")
                
                loss_map = (pred_img -images_high_res).abs()
                # pixel_loss_one_input = loss_map.mean()
                pixel_loss_one_input = loss_map.sum() / num_pixels

                images_high_res = images_high_res.permute(0, 2, 3, 1).squeeze()
                images_low_res = images_low_res.permute(0, 2, 3, 1).squeeze()
                pred_img = pred_img.permute(0, 2, 3, 1).squeeze()
                

                images_high_res = images_high_res.detach().cpu().numpy()
                pred_img = pred_img.detach().cpu().numpy()
                images_low_res = images_low_res.detach().cpu().numpy()

                if args.dataset_select in ["carla", "carla200000"]:

                    if tuple(args.img_size_low_res)[1] != tuple(args.img_size_high_res)[1]:
                        loss_low_res_part = 0
                    else:
                        low_res_index = range(0, h_high_res, downsampling_factor)

                        # Evaluate the loss of low resolution part
                        pred_low_res_part = pred_img[low_res_index, :]
                        loss_low_res_part = np.abs(pred_low_res_part - images_low_res)
                        loss_low_res_part = loss_low_res_part.mean()

                        pred_img[low_res_index, :] = images_low_res

                    # Carla has different projection process as durlar
                    # Refer to code in iln github
                    pred_img = np.flip(pred_img)
                    images_high_res = np.flip(images_high_res)

                    pcd_pred = img_to_pcd_carla(pred_img, maximum_range = 80)
                    pcd_gt = img_to_pcd_carla(images_high_res, maximum_range = 80)
                
                elif args.dataset_select == "kitti":
                    low_res_index = range(0, h_high_res, downsampling_factor)

                    # Evaluate the loss of low resolution part
                    pred_low_res_part = pred_img[low_res_index, :]
                    loss_low_res_part = np.abs(pred_low_res_part - images_low_res)
                    loss_low_res_part = loss_low_res_part.mean()

                    pred_img[low_res_index, :] = images_low_res
                    # 3D Evaluation Metrics
                    pcd_pred = img_to_pcd_kitti(pred_img, maximum_range= 120)
                    pcd_gt = img_to_pcd_kitti(images_high_res, maximum_range = 120)


                elif args.dataset_select == "durlar":
                    # Keep the pixel values in low resolution image
                    low_res_index = range(0, h_high_res, downsampling_factor)

                    # Evaluate the loss of low resolution part
                    pred_low_res_part = pred_img[low_res_index, :]
                    loss_low_res_part = np.abs(pred_low_res_part - images_low_res)
                    loss_low_res_part = loss_low_res_part.mean()

                    pred_img[low_res_index, :] = images_low_res

                    # # Test with edge detector
                    # pred_edges = cv2.Canny((pred_img*255).astype(np.uint8), threshold1=30, threshold2=100)
                    # gt_edges = cv2.Canny((images_high_res*255).astype(np.uint8), threshold1=30, threshold2=100)

                    # 3D Evaluation Metrics
                    pcd_pred = img_to_pcd(pred_img, maximum_range= 120)
                    pcd_gt = img_to_pcd(images_high_res, maximum_range = 120)
                else:
                    raise NotImplementedError(f"Cannot find the dataset: {args.dataset_select}")

                # time_proj = time.time() - start_time
                # print("Time for projection: ", time_proj)

                pcd_all = np.vstack((pcd_pred, pcd_gt))

                chamfer_dist = chamfer_distance(pcd_gt, pcd_pred, num_points = num_pixels)

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
                f1 = 2 * (precision * recall) / (precision + recall)

                # time_metrics = time.time() - time_voxelize
                # print("Time for metrics", time_metrics)



                total_iou[i] += iou
                total_cd[i] += chamfer_dist
                total_loss[i] += pixel_loss_one_input.item()
                total_f1[i] += f1
                total_precision[i] += precision
                total_recall[i] += recall

    evaluation_metrics['mae'] = list(total_loss / global_step)
    evaluation_metrics['chamfer_dist'] = list(total_cd / global_step)
    evaluation_metrics['iou'] = list(total_iou / global_step)
    evaluation_metrics['precision'] = list(total_precision / global_step)
    evaluation_metrics['recall'] = list(total_recall / global_step)
    evaluation_metrics['f1'] = list(total_f1 / global_step)

                # print("Time for one iteration", time.time() - start_time)
                # 
    evaluation_file_path = os.path.join(args.output_dir,'results_different_ranges.txt')
    with open(evaluation_file_path, 'w') as file:
        json.dump(evaluation_metrics, file)

    print(print(f'Dictionary saved to {evaluation_file_path}'))



# TODO: MC Drop
@torch.no_grad()
def MCdrop_diff_ranges(data_loader, model, device, log_writer, args=None):
    # This criterion is also for classfiction, we can directly use the loss forward computation in mae model
    # criterion = torch.nn.CrossEntropyLoss()
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

    if args.dataset_select in ["carla", "carla200000"]:
        iteration_crop_ranges = 8
        max_range = 80
    elif args.dataset_select == "durlar":
        iteration_crop_ranges = 12
        max_range = 120
    elif args.dataset_select == "kitti":
        iteration_crop_ranges = 8
        max_range = 120
    else:
        print("Not Preprocess the pred image")


    grid_size = args.grid_size
    global_step = 0
    total_loss = np.zeros(iteration_crop_ranges)
    total_iou = np.zeros(iteration_crop_ranges)
    total_cd = np.zeros(iteration_crop_ranges)
    total_f1 = np.zeros(iteration_crop_ranges)
    total_precision = np.zeros(iteration_crop_ranges)
    total_recall = np.zeros(iteration_crop_ranges)
    total_frames = np.zeros(iteration_crop_ranges)
    # iterator = iter(data_loader)

    evaluation_metrics = {'mae':[],
                          'chamfer_dist':[],
                          'iou':[],
                          'precision':[],
                          'recall':[],
                          'f1':[]}

    for batch in tqdm.tqdm(data_loader):


        images_low_res = batch[0][0] # (B=1, C, H, W)
        images_high_res = batch[1][0] # (B=1, C, H, W)

        # target = batch[-1]
        images_low_res = images_low_res.to(device, non_blocking=True)
        images_high_res = images_high_res.to(device, non_blocking=True)
        # target = target.to(device, non_blocking=True)
        global_step += 1
        # compute output


        with torch.cuda.amp.autocast():
            
            pred_img_iteration = torch.empty(iteration, images_high_res.shape[1], images_high_res.shape[2], images_high_res.shape[3]).to(device)
            for i in range(int(np.ceil(iteration / iteration_batch))):
                input_batch = iteration_batch if (iteration-i*iteration_batch) > iteration_batch else (iteration-i*iteration_batch)
                test_imgs_input = torch.tile(images_low_res, (input_batch, 1, 1, 1))
                

                pred_imgs = model(test_imgs_input, 
                                images_high_res, 
                                img_size_high_res = args.img_size_high_res,
                                mc_drop = True) # --> tuple(loss, pred_imgs, masked_imgs)
                
                pred_img_iteration[i*iteration_batch:i*iteration_batch+input_batch, ...] = pred_imgs
            pred_img = torch.mean(pred_img_iteration, dim = 0, keepdim = True)
            pred_img_var = torch.std(pred_img_iteration, dim = 0, keepdim = True)
            noise_removal = pred_img_var > noise_threshold * pred_img
            
            pred_img[noise_removal] = 0
        pred_img_full = pred_img.clone()
        images_high_res_full = images_high_res.clone()
        images_low_res_full = images_low_res.clone()
        for i in range(iteration_crop_ranges):
            min_dist = i*10 / max_range
            max_dist = (i+1)*10 / max_range
            range_cropper = CropRanges(min_dist, max_dist)

            pred_img,_= range_cropper(pred_img_full)
            images_high_res, num_pixels = range_cropper(images_high_res_full)
            images_low_res,_ = range_cropper(images_low_res_full)

            if num_pixels != 0:
                total_frames[i] += 1
            else:
                continue

            # loss = criterion(pred_img, images_high_res)
            if log_writer is not None:

                if args.log_transform:
                    # print("Logarithmus Transform back the prediction and ground-truth images")
                    total_loss_one_input = (pred_img - images_high_res).abs().mean()
                    pred_img = torch.expm1(pred_img)
                    images_high_res = torch.expm1(images_high_res)
                    images_low_res = torch.expm1(images_low_res)
                

                    # Preprocess the image
                if args.dataset_select in ["carla", "carla200000"]:
                    pred_img = torch.where((pred_img >= 2/80) & (pred_img <= 1), pred_img, 0)
                elif args.dataset_select == "durlar":
                    pred_img = torch.where((pred_img >= 0.3/120) & (pred_img <= 1), pred_img, 0)
                elif args.dataset_select == "kitti":
                    pred_img = torch.where((pred_img >= 0) & (pred_img <= 1), pred_img, 0)
                else:
                    print("Not Preprocess the pred image")
                
                loss_map = (pred_img -images_high_res).abs()
                pixel_loss_one_input = loss_map.sum() / num_pixels
            
                images_high_res = images_high_res.permute(0, 2, 3, 1).squeeze()
                images_low_res = images_low_res.permute(0, 2, 3, 1).squeeze()
                pred_img = pred_img.permute(0, 2, 3, 1).squeeze()

                

                images_high_res = images_high_res.detach().cpu().numpy()
                pred_img = pred_img.detach().cpu().numpy()
                images_low_res = images_low_res.detach().cpu().numpy()


                if args.dataset_select in ["carla", "carla200000"]:
                    if tuple(args.img_size_low_res)[1] != tuple(args.img_size_high_res)[1]:
                        loss_low_res_part = 0
                    else:
                        low_res_index = range(0, h_high_res, downsampling_factor)

                        # Evaluate the loss of low resolution part
                        pred_low_res_part = pred_img[low_res_index, :]
                        loss_low_res_part = np.abs(pred_low_res_part - images_low_res)
                        loss_low_res_part = loss_low_res_part.mean()

                        pred_img[low_res_index, :] = images_low_res

                    # Carla has different projection process as durlar
                    # Refer to code in iln github
                    pred_img = np.flip(pred_img)
                    images_high_res = np.flip(images_high_res)
                    pcd_pred = img_to_pcd_carla(pred_img, maximum_range = 80)
                    pcd_gt = img_to_pcd_carla(images_high_res, maximum_range = 80)

                elif args.dataset_select == "kitti":
                    low_res_index = range(0, h_high_res, downsampling_factor)

                    # Evaluate the loss of low resolution part
                    pred_low_res_part = pred_img[low_res_index, :]
                    loss_low_res_part = np.abs(pred_low_res_part - images_low_res)
                    loss_low_res_part = loss_low_res_part.mean()

                    pred_img[low_res_index, :] = images_low_res
                    # 3D Evaluation Metrics
                    pcd_pred = img_to_pcd_kitti(pred_img, maximum_range= 120)
                    pcd_gt = img_to_pcd_kitti(images_high_res, maximum_range = 120)

                elif args.dataset_select == "durlar":
                    # Keep the pixel values in low resolution image
                    low_res_index = range(0, h_high_res, downsampling_factor)

                    # Evaluate the loss of low resolution part
                    pred_low_res_part = pred_img[low_res_index, :]
                    loss_low_res_part = np.abs(pred_low_res_part - images_low_res)
                    loss_low_res_part = loss_low_res_part.mean()

                    pred_img[low_res_index, :] = images_low_res
                    
                    # 3D Evaluation Metrics
                    pcd_pred = img_to_pcd(pred_img)
                    pcd_gt = img_to_pcd(images_high_res)
                
                else:
                    raise NotImplementedError(f"Cannot find the dataset: {args.dataset_select}")

                pcd_all = np.vstack((pcd_pred, pcd_gt))

                chamfer_dist = chamfer_distance(pcd_gt, pcd_pred, num_points = num_pixels)
                min_coord = np.min(pcd_all, axis=0)
                max_coord = np.max(pcd_all, axis=0)
                
                # Voxelize the ground truth and prediction point clouds
                voxel_grid_predicted = voxelize_point_cloud(pcd_pred, grid_size, min_coord, max_coord)
                voxel_grid_ground_truth = voxelize_point_cloud(pcd_gt, grid_size, min_coord, max_coord)
                # Calculate metrics
                iou, precision, recall = calculate_metrics(voxel_grid_predicted, voxel_grid_ground_truth)

                f1 = 2 * (precision * recall) / (precision + recall)

                total_iou[i] += iou
                total_cd[i] += chamfer_dist
                total_loss[i] += pixel_loss_one_input.item()
                total_f1[i] += f1
                total_precision[i] += precision
                total_recall[i] += recall

    evaluation_metrics['mae'] = list(total_loss / total_frames)
    evaluation_metrics['chamfer_dist'] = list(total_cd / total_frames)
    evaluation_metrics['iou'] = list(total_iou / total_frames)
    evaluation_metrics['precision'] = list(total_precision / total_frames)
    evaluation_metrics['recall'] = list(total_recall / total_frames)
    evaluation_metrics['f1'] = list(total_f1 / total_frames)

    evaluation_file_path = os.path.join(args.output_dir,'evaluate_different_ranges_mcdrop.txt')
    with open(evaluation_file_path, 'w') as file:
        json.dump(evaluation_metrics, file) 

    print(print(f'Dictionary saved to {evaluation_file_path}'))



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

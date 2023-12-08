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

from util.datasets import grid_reshape, grid_reshape_backward
import tqdm

from einops import rearrange

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

def patchify(imgs, patch_size, grid_size, in_chans = 1):
    """
    imgs: (N, 3, H, W)
    x: (N, L, patch_size**2 *3)
    """
    ph, pw = patch_size
    h, w = grid_size
    # assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

    x = imgs.reshape(shape=(imgs.shape[0], in_chans, h, ph, w, pw))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(imgs.shape[0], h * w, ph * pw * in_chans)
    return x

def unpatchify(x, patch_size, grid_size, in_chans = 1):
    """
    x: (N, L, patch_size**2 *3)
    imgs: (N, 3, H, W)
    """
    # p = self.patch_size
    # h = w = int(x.shape[1] ** .5)
    ph, pw = patch_size
    h, w = grid_size

    assert h * w == x.shape[1]
    x = x.reshape(shape=(x.shape[0], h, w, ph, pw, in_chans))
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(x.shape[0], in_chans, h * ph, w * pw)
    return imgs


@torch.no_grad()
def analyze(data_loader, model, device, log_writer, args=None):
    # This criterion is also for classfiction, we can directly use the loss forward computation in mae model
    # criterion = torch.nn.CrossEntropyLoss()

    # metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    img_size_high_res = tuple(args.img_size_high_res)


    # switch to evaluation mode
    model.eval()
    global_step = 0
    local_step = 0
    window_size = args.window_size
    patch_size = args.patch_size

    # iterator = iter(data_loader)

    for batch in tqdm.tqdm(data_loader):
        if global_step != 899:
            global_step += 1
            continue
        images_low_res = batch[0][0] # (B=1, C, H, W)
        images_high_res = batch[1][0] # (B=1, C, H, W)

        
        # target = batch[-1]
        images_low_res = images_low_res.to(device, non_blocking=True)
        images_high_res = images_high_res.to(device, non_blocking=True)

        #######
        images_low_res_save = torch.expm1(images_low_res)
        images_low_res_save = images_low_res_save.permute(0, 2, 3, 1).squeeze()
        images_low_res_save = images_low_res_save.detach().cpu().numpy()
        # images_low_res_save = scalarMap.to_rgba(images_low_res_save)[..., :3]
        np.save('test_img.npy', images_low_res_save)

        # target = target.to(device, non_blocking=True)
        
        global_step += 1
        
        # compute output

        with torch.cuda.amp.autocast():
            output = model(images_low_res, 
                                    images_high_res, 
                                    img_size_high_res = args.img_size_high_res,
                                    eval = True) # --> tuple(loss, pred_imgs, masked_imgs)

        # Just for debugging    
        # if global_step != 1:
        #     break
            
        if log_writer is not None:
            # Visulize less for carla dataset

            # Preprocess the image

            # if global_step % 100 != 0 and global_step != 1:
            #     continue

            pred_img = output['pred']
            
            
            loss_map = (pred_img -images_high_res).abs()
            loss_map_normalized = (loss_map - loss_map.min()) / (loss_map.max() - loss_map.min() + 1e-8)
            loss_map_normalized = loss_map_normalized.permute(0, 2, 3, 1).squeeze()
            loss_map_normalized = loss_map_normalized.detach().cpu().numpy()
            loss_map_normalized = scalarMap_loss_map.to_rgba(loss_map_normalized)[..., :3]


            images_high_res = images_high_res.permute(0, 2, 3, 1).squeeze()
            images_high_res = images_high_res.detach().cpu().numpy()
            images_high_res = scalarMap.to_rgba(images_high_res)[..., :3]


            


            log_writer.add_image('loss_map', torch.Tensor(loss_map_normalized).permute(2, 0, 1), local_step)
            log_writer.add_image('gt-highres', torch.Tensor(images_high_res).permute(2, 0, 1), local_step)

            feature_map_downsample = output['features_downsample']
            feature_map_upsample = output['features_upsample']
            # pretrain_feature_map_downsample = output['pretrain_downsample_features']

            # feature_lowres_backbone = output['feature_backbone']
            # feature_highres_backbone = output['target_feature_backbone']
            # cosine_similarity_map = output['cosine_similarity_map']

            num_grids = img_size_high_res[1] // img_size_high_res[0]

            feature_map_first = feature_map_downsample[0]
            feature_map_last = feature_map_upsample[-1] 
            input_size = (args.img_size_low_res)
            h_in = input_size[0] // patch_size[0]
            w_in = input_size[1] // patch_size[1]

            feature_map_first = feature_map_first.sum(dim=-1, keepdim=True)
            feature_map_last = feature_map_last.sum(dim=-1, keepdim=True)
            feature_map_first = (feature_map_first - feature_map_first.min()) / (feature_map_first.max() - feature_map_first.min() + 1e-8)
            feature_map_last = (feature_map_last - feature_map_last.min()) / (feature_map_last.max() - feature_map_last.min() + 1e-8)
            if window_size[0] == window_size[1]:
                feature_map_first= feature_map_first.reshape(1, h_in, w_in, 1)
                feature_map_last = feature_map_last.reshape(1, img_size_high_res[0], img_size_high_res[1], 1)

            feature_map_first = feature_map_first.squeeze() # H*W
            feature_map_first = feature_map_first.detach().cpu().numpy()

            feature_map_last = feature_map_last.squeeze() # H*W
            feature_map_last = feature_map_last.detach().cpu().numpy()

            feature_map_first = scalarMap_loss_map.to_rgba(feature_map_first)[..., :3]
            feature_map_last = scalarMap_loss_map.to_rgba(feature_map_last)[..., :3]


            log_writer.add_image(f'feature_map_first', torch.Tensor(feature_map_first).permute(2, 0, 1), local_step)
            log_writer.add_image(f'feature_map_last', torch.Tensor(feature_map_last).permute(2, 0, 1), local_step)

            
            # for i, feature_map in enumerate(feature_map_downsample):
            #     # print("Downsample")
            #     # Grid Reshaping Backward: (512x512 -> 128 x 2048)
            #     # print(feature_map.shape)
            #     b, h, w, c = feature_map.shape
            #     feature_map = feature_map.sum(dim=-1, keepdim=True) # b h w c

            #     feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-8)

            #     h_in = int((h*w / num_grids) ** 0.5)
            #     w_in = int(h*w / h_in)

            #     # # print(h_in, w_in)
            #     # params = (h_in, 
            #     #             w_in,
            #     #             1,
            #     #             num_grids,
            #     #             int(num_grids**0.5))
                

            #     # feature_map = grid_reshape_backward(feature_map, params, order="bhwc")
            #     if window_size[0] == window_size[1]:
            #         feature_map = feature_map.reshape(b, h_in, w_in, 1)

            #     feature_map = feature_map.squeeze() # H*W
            #     feature_map = feature_map.detach().cpu().numpy()

            #     feature_map = scalarMap_loss_map.to_rgba(feature_map)[..., :3]

            #     log_writer.add_image(f'Downsample:{str(h)}-{str(w)}-{str(c)}-{str(i)}', torch.Tensor(feature_map).permute(2, 0, 1), local_step)


            # for i, feature_map in enumerate(feature_map_upsample):
            #     # print("Upsample")
            #     # Grid Reshaping Backward: (512x512 -> 128 x 2048)
            #     # print(feature_map.shape)
            #     b, h, w, c = feature_map.shape
            #     feature_map = feature_map.sum(dim=-1, keepdim=True) # b h w c

            #     feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-8)
                
            #     h_in = int((h*w / num_grids) ** 0.5)
            #     w_in = int(h*w / h_in)

            #     # print(h_in, w_in)
            #     # params = (h_in, 
            #     #             w_in,
            #     #             1,
            #     #             num_grids,
            #     #             int(num_grids**0.5))
                

            #     # feature_map = grid_reshape_backward(feature_map, params, order="bhwc")

            #     if window_size[0] == window_size[1]:
            #         feature_map = feature_map.reshape(b, h_in, w_in, 1)


            #     feature_map = feature_map.squeeze() # H*W
            #     feature_map = feature_map.detach().cpu().numpy()

            #     feature_map = scalarMap_loss_map.to_rgba(feature_map)[..., :3]

            #     log_writer.add_image(f'Upsample:{str(h)}-{str(w)}-{str(c)}-{str(i)}', torch.Tensor(feature_map).permute(2, 0, 1), local_step)

            

            local_step += 1

            
            
        
    # # results = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    # avg_losses = total_losses / global_step
    # if log_writer is not None:

    #     log_writer.add_scalar('Average/l1_loss', avg_losses[0], 0)
    #     log_writer.add_scalar('Average/relative_loss(row, 20 neighbour pixels)', avg_losses[1], 0)
    #     log_writer.add_scalar('Average/relative_loss(row, 500 neighbour pixels)', avg_losses[2], 0)
    #     log_writer.add_scalar('Average/relative_loss(square, 20 neighbour pixels, dilation=0)', avg_losses[3], 0)
    #     log_writer.add_scalar('Average/relative_loss(square, 500 neighbour pixels, dilation=0)', avg_losses[4], 0)
    #     log_writer.add_scalar('Average/relative_loss(square, 20 neighbour pixels, dilation=1)', avg_losses[5], 0)
    #     log_writer.add_scalar('Average/relative_loss(square, 500 neighbour pixels, dilation=1)', avg_losses[6], 0)





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

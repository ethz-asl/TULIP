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

cNorm = colors.Normalize(vmin=0, vmax=1)
jet = plt.get_cmap('viridis_r')
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)


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
    # iterator = iter(data_loader)
    for batch in data_loader:
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
            
            # if not args.reverse_pixel_value:
            #     # If not revers the pixel value in input data, then make it for visualization
            #     mask_for_masked_image = masked_img == 1

            #     pred_img = 1 - pred_img
            #     pred_img[pred_img == 1] = 0

            #     images = 1 - images
            #     images[images == 1] = 0

            #     masked_img = 1 - masked_img
            #     masked_img[masked_img == 1] = 0
            #     masked_img[mask_for_masked_image] = 1


            # loss = criterion(output[1], target)
        if log_writer is not None:
            if args.use_intensity and args.in_chans == 2:
                vis_grid = make_grid([images[:, 0], pred_img[:, 0], masked_img[:, 0],
                                    images[:, 1], pred_img[:, 1], masked_img[:, 1]], nrow=3, ncol=2)
                log_writer.add_image('depth+intensity: gt - pred - mask', vis_grid, global_step)
            else:
                images = images.permute(0, 3, 1, 2).squeeze()
                pred_img = pred_img.permute(0, 3, 1, 2).squeeze()
                masked_img = masked_img.permute(0, 3, 1, 2).squeeze()
                images = images.detach().cpu().numpy()
                pred_img = pred_img.detach().cpu().numpy()
                masked_img = masked_img.detach().cpu().numpy()

                images = scalarMap.to_rgba(images)[..., :3]
                pred_img = scalarMap.to_rgba(pred_img)[..., :3]
                mask_for_masked_image = masked_img == 1
                mask_for_masked_image = mask_for_masked_image.squeeze()  
                masked_img = scalarMap.to_rgba(masked_img)[..., :3]
                masked_img[mask_for_masked_image,:] = [1, 1, 1]

                # vis_grid = make_grid(torch.cat([images, pred_img, masked_img], dim = 0), nrow=1)
                vis_grid = make_grid([torch.Tensor(images).permute(2, 1, 0), torch.Tensor(pred_img).permute(2, 1, 0), torch.Tensor(masked_img).permute(2, 1, 0)], nrow=1)
                log_writer.add_image('gt - pred - mask', vis_grid, global_step)
            log_writer.add_scalar('Loss/test', loss.item(), global_step)
            
        total_loss += loss.item()
    
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

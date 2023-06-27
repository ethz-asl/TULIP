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
jet = plt.get_cmap('magma')
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    criterion:  torch.nn.L1Loss(), 
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
            pred = model(samples_low_res.contiguous())

        loss = criterion(pred, samples_high_res.contiguous())
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
def evaluate(data_loader, model, device, log_writer, criterion = torch.nn.L1Loss(), args=None):
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
        images_low_res = batch[0][0] # (B=1, C, H, W)
        images_high_res = batch[1][0] # (B=1, C, H, W)

        # target = batch[-1]
        images_low_res = images_low_res.to(device, non_blocking=True)
        images_high_res = images_high_res.to(device, non_blocking=True)
        # target = target.to(device, non_blocking=True)
        global_step += 1
        # compute output
        with torch.cuda.amp.autocast():
            pred_img = model(images_low_res) # --> tuple(loss, pred_imgs, masked_imgs)
        loss = criterion(pred_img, images_high_res)
            
        if log_writer is not None:
            images_high_res = images_high_res.permute(0, 3, 1, 2).squeeze()
            pred_img = pred_img.permute(0, 3, 1, 2).squeeze()

            images_high_res = images_high_res.detach().cpu().numpy()
            pred_img = pred_img.detach().cpu().numpy()

            images_high_res = scalarMap.to_rgba(images_high_res)[..., :3]
            pred_img = scalarMap.to_rgba(pred_img)[..., :3]

            # vis_grid = make_grid(torch.cat([images, pred_img, masked_img], dim = 0), nrow=1)
            vis_grid = make_grid([torch.Tensor(images_high_res).permute(2, 1, 0), torch.Tensor(pred_img).permute(2, 1, 0)], nrow=1)
            log_writer.add_image('gt - pred', vis_grid, global_step)
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

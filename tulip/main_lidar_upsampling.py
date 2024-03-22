# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from util.datasets import generate_dataset
from util.pos_embed import interpolate_pos_embed

import timm.optim.optim_factory as optim_factory
from timm.models.layers import trunc_normal_

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import model.tulip as tulip
from engine_upsampling import train_one_epoch, evaluate, get_latest_checkpoint, MCdrop
import wandb


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    

    # Model parameters
    parser.add_argument('--model_select', default='mae', type=str,
                        choices=['tulip_base', 'tulip_large'])

    parser.add_argument('--window_size', nargs="+", type=int,
                        help='size of window partition')
    parser.add_argument('--remove_mask_token', action="store_true", help="Remove mask token in the encoder")
    parser.add_argument('--patch_size', nargs="+", type=int, help='image size, given in format h w')
    
    parser.add_argument('--pixel_shuffle', action='store_true',
                        help='pixel shuffle upsampling head')
    parser.add_argument('--circular_padding', action='store_true',
                        help='circular padding, kernel size is 1, 8 and stride is 1, 4')
    parser.add_argument('--patch_unmerging', action='store_true',
                        help='reverse operation of patch merging')
    parser.add_argument('--swin_v2', action='store_true',
                        help='use swin_v2 block')

    
    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')
    
    
    # Augmentation parameters
    parser.add_argument('--roll', action="store_true", help='random roll range map in vertical direction')

    # Dataset parameters
    parser.add_argument('--dataset_select', default='durlar', type=str, choices=['durlar', 'carla','kitti'])
    parser.add_argument('--img_size_low_res', nargs="+", type=int, help='low resolution image size, given in format h w')
    parser.add_argument('--img_size_high_res', nargs="+", type=int, help='high resolution image size, given in format h w')
    parser.add_argument('--in_chans', type=int, default = 1, help='number of channels')
    parser.add_argument('--data_path_low_res', default=None, type=str,
                        help='low resolution dataset path')
    parser.add_argument('--data_path_high_res', default=None, type=str,
                        help='high resolution dataset path')
    
    parser.add_argument('--save_pcd', action="store_true", help='save pcd output in evaluation step')
    parser.add_argument('--log_transform', action="store_true", help='apply log1p transform to data')
    parser.add_argument('--keep_close_scan', action="store_true", help='mask out pixel belonging to further object')
    parser.add_argument('--keep_far_scan', action="store_true", help='mask out pixel belonging to close object')
    

    # Training parameters
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--save_frequency', default=100, type=int,help='frequency of saving the checkpoint')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    
    
    # Logger parameters
    parser.add_argument('--wandb_disabled', action='store_true', help="disable wandb")
    parser.add_argument('--entity', type = str, default = "biyang")
    parser.add_argument('--project_name', type = str, default = "Ouster_MAE")
    parser.add_argument('--run_name', type = str, default = None)

    # Evaluation parameters
    parser.add_argument('--eval', action='store_true', help="evaluation")
    parser.add_argument('--mc_drop', action='store_true', help="apply monte carlo dropout at inference time")
    parser.add_argument('--num_mcdropout_iterations', type = int, default=50, help="number of inference for monte carlo dropout")
    parser.add_argument('--noise_threshold', type = float, default=0.03, help="threshold of monte carlo dropout")
    parser.add_argument('--grid_size', type = float, default=0.1, help="grid size for voxelization")
    
    
    return parser


def main(args):
    
    misc.init_distributed_mode(args)
    


    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    dataset_train = generate_dataset(is_train = True, args = args)
    dataset_val = generate_dataset(is_train = False, args = args)

    print(f"There are totally {len(dataset_train)} training data and {len(dataset_val)} validation data")


    
    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()

        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        
        # Validation set uses only one rank to write the summary
        sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    # Logger is only used in one rank
    if global_rank == 0:
        if args.wandb_disabled:
            mode = "disabled"
        else:
            mode = "online"
        wandb.init(project=args.project_name,
                    entity=args.entity,
                    name = args.run_name, 
                    mode=mode,
                    sync_tensorboard=True)
        wandb.config.update(args)
    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )


    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )
    
    
    # define the model
    model = tulip.__dict__[args.model_select](img_size = tuple(args.img_size_low_res),
                                                    target_img_size = tuple(args.img_size_high_res),
                                                        patch_size = tuple(args.patch_size), 
                                                        in_chans = args.in_chans,
                                                        window_size = args.window_size,
                                                        swin_v2 = args.swin_v2,
                                                        pixel_shuffle = args.pixel_shuffle,
                                                        circular_padding = args.circular_padding,
                                                        log_transform = args.log_transform,
                                                        patch_unmerging = args.patch_unmerging)
    

    if args.eval and os.path.exists(args.output_dir):
        print("Loading Checkpoint and directly start the evaluation")
        if args.output_dir.endswith("pth"):
            args.resume = args.output_dir
            args.output_dir = os.path.dirname(args.output_dir)
        else:
            get_latest_checkpoint(args)

        misc.load_model(
                args=args, model_without_ddp=model, optimizer=None,
                loss_scaler=None)
        model.to(device)
        
        print("Start Evaluation")
        if args.mc_drop:
            print("Evaluation with Monte Carlo Dropout")
            MCdrop(data_loader_val, model, device, log_writer = log_writer, args = args)
        else:
            evaluate(data_loader_val, model, device, log_writer = log_writer, args = args)
        print("Evaluation finished")


        exit(0)
    else:
        print("Start Training")
        

    model.to(device)

    model_without_ddp = model
    # print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module
    
    # following timm: set wd as 0 for bias and norm layers
    # param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    param_groups = optim_factory.param_groups_layer_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()
    

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )
        if args.output_dir and (epoch % args.save_frequency == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    print('Training finished')

    if global_rank == 0:
        wandb.finish()
if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    if args.output_dir and not args.eval:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
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
from util.datasets import build_durlar_upsampling_dataset, build_carla_upsampling_dataset, build_kitti_upsampling_dataset
from util.pos_embed import interpolate_pos_embed

import timm

# assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory
from timm.models.layers import trunc_normal_

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

# import swin_unet_template as swin_unet
import swin_unet
import swinunet_modelanalysis
from timm.data.dataset import ImageDataset
from engine_upsampling import train_one_epoch, evaluate, get_latest_checkpoint, MCdrop
from engine_upsampling_analysis import analyze
import wandb

import swin_mae_ploss


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')


    # Pretrain parameters (MAE)
    parser.add_argument('--pretrain', default=None, type=str,
                        help='full path of pretrain model')
    parser.add_argument('--pretrain_only_encoder', action='store_true', help="Only load pretrained weights of the encoder")
    parser.add_argument('--use_cls_token', action='store_true', help="Use CLS token in embedding")

    # Swin MAE parameters
    parser.add_argument('--window_size', default=7, type=int,
                        help='size of window partition')
    parser.add_argument('--remove_mask_token', action="store_true", help="Remove mask token in the encoder")
    parser.add_argument('--patch_size', nargs="+", type=int, help='image size, given in format h w')


    # Model parameters
    parser.add_argument('--model_select', default='mae', type=str,
                        choices=['mae', 'swin_mae', 'swin_unet', 
                                 'swin_unet_v2', 'swin_unet_deep', 'swin_unet_deeper',
                                 'swin_unet_moredepths', 'swin_unet_v2_deep', 'swin_unet_v2_deeper',
                                 'swin_unet_v2_moredepths', 'swin_unet_with_pretrain'])

    parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    
    parser.add_argument('--pretrain_mae_model', default='swin_mae_patch2_base_line_ws4', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    
    parser.add_argument('--edge_loss', action='store_true',
                        help='Add edge loss')
    
    parser.add_argument('--depth_scale_loss', action='store_true',
                        help='scale up the loss with ground truth depth values')
    
    parser.add_argument('--relative_dist_loss', action="store_true", help='relative distance loss')

    parser.add_argument('--perceptual_loss', action="store_true", help='perceptual loss function (locked by pretrained MAE)')
    
    parser.add_argument('--pixel_shuffle', action='store_true',
                        help='pixel shuffle upsampling head')
    parser.add_argument('--grid_reshape', action='store_true',
                        help='grid reshape image')
    parser.add_argument('--circular_padding', action='store_true',
                        help='circular padding, kernel size is 1, 8 and stride is 1, 4')
    parser.add_argument('--pixel_shuffle_expanding', action='store_true',
                        help='pixel shuffle upsampling in expanding path')
    

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)
    
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
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    
     # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # Dataset parameters
    parser.add_argument('--dataset_select', default='durlar', type=str, choices=['durlar', 'carla','kitti', 'image-net'])

    parser.add_argument('--gray_scale', action="store_true", help='use gray scale imgae')
    parser.add_argument('--img_size_low_res', nargs="+", type=int, help='low resolution image size, given in format h w')
    parser.add_argument('--img_size_high_res', nargs="+", type=int, help='high resolution image size, given in format h w')
    parser.add_argument('--in_chans', type=int, default = 1, help='number of channels')
    parser.add_argument('--data_path_low_res', default=None, type=str,
                        help='low resolution dataset path')
    parser.add_argument('--data_path_high_res', default=None, type=str,
                        help='high resolution dataset path')
    
    
    # Durlar dataset parameters
    parser.add_argument('--crop', action="store_true", help='crop the image to 128 x 128 (default)')
    parser.add_argument('--mask_loss', action="store_true", help='Mask the loss value with no LiDAR return')
    parser.add_argument('--use_intensity', action="store_true", help='use the intensity as the second channel')
    parser.add_argument('--reverse_pixel_value', action="store_true", help='reverse the pixel value in the input')
    parser.add_argument('--save_pcd', action="store_true", help='save pcd output in evaluation step')
    parser.add_argument('--log_transform', action="store_true", help='apply log1p transform to data')
    parser.add_argument('--keep_close_scan', action="store_true", help='mask out pixel belonging to further object')
    parser.add_argument('--keep_far_scan', action="store_true", help='mask out pixel belonging to close object')

    parser.add_argument('--roll', action="store_true", help='random roll range map in vertical direction')
    
    
    

    # Training parameters
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
    
    
    parser.add_argument('--wandb_disabled', action='store_true', help="disable wandb")
    parser.add_argument('--entity', type = str, default = "biyang")
    parser.add_argument('--project_name', type = str, default = "Ouster_MAE")
    parser.add_argument('--run_name', type = str, default = None)

    parser.add_argument('--eval', action='store_true', help="evaluation")
    parser.add_argument('--analyze', action='store_true', help="call model analysis")
    parser.add_argument('--mc_drop', action='store_true', help="apply monte carlo dropout at inference time")
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

    if args.dataset_select == 'durlar':
        dataset_train = build_durlar_upsampling_dataset(is_train = True, args = args)
        dataset_val = build_durlar_upsampling_dataset(is_train = False, args = args)
    elif args.dataset_select == 'carla':
        dataset_train = build_carla_upsampling_dataset(is_train = True, args = args)
        dataset_val = build_carla_upsampling_dataset(is_train = False, args = args)
    elif args.dataset_select == 'kitti':
        dataset_train = build_kitti_upsampling_dataset(is_train = True, args = args)
        dataset_val = build_kitti_upsampling_dataset(is_train = False, args = args)
    else:
        raise NotImplementedError("Cannot find the matched dataset builder")
    
    
    
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

    if args.perceptual_loss and not (args.eval or args.analyze):
        print("Unlocking MAE perceptual loss")

        pretrain_mae_model = swin_mae_ploss.__dict__[args.pretrain_mae_model](
                                                                            # img_size = tuple(args.img_size_low_res),
                                                                            img_size = tuple(args.img_size_high_res),
                                                                            norm_pix_loss=args.norm_pix_loss,
                                                                            in_chans = args.in_chans,
                                                                            circular_padding = args.circular_padding,
                                                                            grid_reshape = args.grid_reshape,
                                                                            conv_projection = args.pixel_shuffle,
                                                                            pixel_shuffle_expanding = args.pixel_shuffle_expanding,
                                                                            log_transform = args.log_transform,)

        pretrain = torch.load(args.pretrain, map_location='cpu')

        print("Load pre-trained MAE checkpoint from: %s" % args.pretrain)
        pretrain_model = pretrain['model']

        msg = pretrain_mae_model.load_state_dict(pretrain_model, strict=False)
        print(msg)

        print("Freeze the parameters in pretrained mae model")
        for name, p in pretrain_mae_model.named_parameters():
            p.requires_grad = False

        pretrain_mae_model.to(device)

    elif args.perceptual_loss and args.eval:
        pretrain_mae_model = swin_mae_ploss.__dict__[args.pretrain_mae_model](img_size = tuple(args.img_size_high_res),
                                                                            norm_pix_loss=args.norm_pix_loss,
                                                                            in_chans = args.in_chans,
                                                                            circular_padding = args.circular_padding,
                                                                            grid_reshape = args.grid_reshape,
                                                                            conv_projection = args.pixel_shuffle,
                                                                            pixel_shuffle_expanding = args.pixel_shuffle_expanding,
                                                                            log_transform = args.log_transform,)
        pretrain_mae_model.to(device)
    else:
        pretrain_mae_model = None

        


    if args.analyze:
        if args.perceptual_loss:
            pretrain_mae_model = swinunet_modelanalysis.__dict__['swin_mae_analysis_for_pretrain'](img_size = tuple(args.img_size_high_res),
                                                                            norm_pix_loss=args.norm_pix_loss,
                                                                            circular_padding = args.circular_padding,
                                                                            grid_reshape = args.grid_reshape,
                                                                            conv_projection = args.pixel_shuffle,
                                                                            pixel_shuffle_expanding = args.pixel_shuffle_expanding,
                                                                            log_transform = args.log_transform,)
            
            if args.pretrain is not None:
                pretrain = torch.load(args.pretrain, map_location='cpu')

                print("Load pre-trained MAE checkpoint from: %s" % args.pretrain)
                pretrain_model = pretrain['model']

                msg = pretrain_mae_model.load_state_dict(pretrain_model, strict=False)
                print(msg)

                print("Freeze the parameters in pretrained mae model")
                for name, p in pretrain_mae_model.named_parameters():
                    p.requires_grad = False

                pretrain_mae_model.to(device)

        else:
            pretrian_mae_model = None

        model = swinunet_modelanalysis.__dict__[args.model_select](img_size = tuple(args.img_size_low_res),
                                                    target_img_size = tuple(args.img_size_high_res),
                                                        patch_size = tuple(args.patch_size), 
                                                        in_chans = args.in_chans,
                                                        window_size = args.window_size,
                                                        edge_loss = args.edge_loss,
                                                        pixel_shuffle = args.pixel_shuffle,
                                                        grid_reshape = args.grid_reshape,
                                                        circular_padding = args.circular_padding,
                                                        log_transform = args.log_transform,
                                                        depth_scale_loss = args.depth_scale_loss,
                                                        pixel_shuffle_expanding = args.pixel_shuffle_expanding,
                                                        relative_dist_loss = args.relative_dist_loss,
                                                        perceptual_loss = args.perceptual_loss,
                                                        pretrain_mae = pretrain_mae_model,)


    else:
        model = swin_unet.__dict__[args.model_select](img_size = tuple(args.img_size_low_res),
                                                    target_img_size = tuple(args.img_size_high_res),
                                                        patch_size = tuple(args.patch_size), 
                                                        in_chans = args.in_chans,
                                                        window_size = args.window_size,
                                                        edge_loss = args.edge_loss,
                                                        pixel_shuffle = args.pixel_shuffle,
                                                        grid_reshape = args.grid_reshape,
                                                        circular_padding = args.circular_padding,
                                                        log_transform = args.log_transform,
                                                        depth_scale_loss = args.depth_scale_loss,
                                                        pixel_shuffle_expanding = args.pixel_shuffle_expanding,
                                                        relative_dist_loss = args.relative_dist_loss,
                                                        perceptual_loss = args.perceptual_loss,
                                                        pretrain_mae = pretrain_mae_model,)
        
    # Load pretrained model
    if args.pretrain is not None and not args.perceptual_loss:
        pretrain = torch.load(args.pretrain, map_location='cpu')

        print("Load pre-trained checkpoint from: %s" % args.pretrain)
        pretrain_model = pretrain['model']

        state_dict = model.state_dict()
        for k in ['head.weight', 'patch_embed.proj.weight']:
            if k in pretrain_model and pretrain_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del pretrain_model[k]

        
        # state_dict = model.state_dict()

        # # if args.pretrain_only_encoder:
        # print(pretrain_model.keys())

        if args.pretrain_only_encoder:
            for k in list(pretrain_model.keys()):
                if k.__contains__('up'):
                    print(f"Removing key {k} from pretrained checkpoint")
                    del pretrain_model[k]
            
        msg = model.load_state_dict(pretrain_model, strict=False)
        print(msg)

        # # Freeze the parameters in pretrain model
        # for name, p in model.named_parameters():
        #     if name in list(pretrain_model.keys()):
        #         p.requires_grad = False

    else:
        print("Training from scratch")

    if args.eval and os.path.exists(args.output_dir):
        print("Loading Checkpoint and directly start the evaluation")
        if args.output_dir.endswith("pth"):
            args.resume = args.output_dir
            args.output_dir = os.path.dirname(args.output_dir)
        else:
            get_latest_checkpoint(args)
        # checkpoint = torch.load(args.resume, map_location='cpu')
        # model.load_state_dict(checkpoint)
        misc.load_model(
                args=args, model_without_ddp=model, optimizer=None,
                loss_scaler=None)
        model.to(device)
        
        print("Start Evaluation")
        if args.mc_drop:
            print("Apply Monte-Carlo Dropout")
            MCdrop(data_loader_val, model, device, log_writer = log_writer, args = args)
        else:
            evaluate(data_loader_val, model, device, log_writer = log_writer, args = args)
        print("Evaluation finished")


        exit(0)

    elif args.analyze and os.path.exists(args.output_dir):
        print("Loading Checkpoint and directly start the evaluation")
        get_latest_checkpoint(args)

        if args.pretrain is not None:
            checkpoint = torch.load(args.resume, map_location='cpu')
            checkpoint_model = checkpoint['model']

            for k in list(checkpoint_model.keys()):
                if k.__contains__('pretrain_mae'):
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]


            model.load_state_dict(checkpoint_model, strict = False)
        else:

            misc.load_model(
                    args=args, model_without_ddp=model, optimizer=None,
                    loss_scaler=None)
        model.to(device)

        print("Start Model Analysis")
        analyze(data_loader_val, model, device, log_writer = log_writer, args = args)

        print("Analysis finished")
        exit(0)
        

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
        if args.output_dir and (epoch % 100 == 0 or epoch + 1 == args.epochs):
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

    # print("Start Evaluation")
    # if args.mc_drop:
    #     print("Apply Monte-Carlo Dropout")
    #     MCdrop(data_loader_val, model, device, log_writer = log_writer, args = args)
    # else:
    #     evaluate(data_loader_val, model, device, log_writer = log_writer, args = args)
    # print("Evaluation finished")

    if global_rank == 0:
        wandb.finish()
if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    # if args.img_size is not None:
    #     args.img_size = tuple(args.img_size)

    # Pre-Check
    if args.use_intensity:
        assert args.in_chans > 1
    if args.output_dir and not args.eval:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

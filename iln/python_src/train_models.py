import argparse
import yaml
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm

# Datasets
from dataset.range_images_dataset import RangeImagesDataset
from dataset.samples_from_image_dataset import SamplesFromImageDataset
from dataset.dataset_utils import generate_dataset

# Models
from models.iln.iln import ILN
from models.liif_cvpr21.liif_lidar import LIIFLiDAR
from models.lsr_ras20.unet import UNet
from models.model_utils import generate_model

import wandb




def is_valid_check_point():
    if check_point['model']['name'] != config['model']['name']:
        return False

    for key, value in check_point['model']['args'].items():
        if value != config['model']['args'][key]:
            return False

    if check_point['lidar_in'] != train_dataset.lidar_in:
        return False

    return True


def print_log(epoch, loss_sum, loss_avg, directory=None):
    log_msg = ('%03d %.4f %.4f' % (epoch, loss_sum, loss_avg))
    with open(os.path.join(directory, 'training_loss_history.txt'), 'a') as f:
        f.write(log_msg + '\n')

    return print(log_msg)


def save_check_point(epoch, period=10):
    if epoch % period == period-1:
        check_point_model_info = {'name': config['model']['name'],
                                  'args': config['model']['args'],
                                  'state_dict': model.state_dict() if n_gpus <= 1 else model.module.state_dict()}
        check_point = {'epoch': epoch + 1,
                       'model': check_point_model_info,
                       'optimizer': optimizer.state_dict(),
                       'lr_scheduler': lr_scheduler.state_dict(),
                       'lidar_in': train_dataset.lidar_in}

        check_point_filename = os.path.join(model_directory, model_name + '_' + str(epoch + 1) + '.pth')
        torch.save(check_point, check_point_filename)

    return


def train_implicit_network():
    for epoch in range(epoch_start, epoch_end, 1):
        loss_sum = 0.0

        for input_range_images, input_queries, output_ranges in tqdm(train_loader, leave=False, desc='train'):
            # Load data: [-1 ~ 1]
            input_range_images = input_range_images.cuda()
            input_queries = input_queries.cuda()

            # Initialize gradient
            optimizer.zero_grad()

            # Prediction
            pred_ranges = model(input_range_images, input_queries)

            # Loss
            output_ranges = output_ranges.cuda()
            loss = criterion(pred_ranges, output_ranges)

            # Back-propagation
            loss.backward()
            optimizer.step()

            loss_sum = loss_sum + loss.detach().cpu()

            wandb.log({'Train/loss': loss.item()})

        # Schedule the learning rate
        lr_scheduler.step()

        # Logging
        print_log(epoch, loss_sum, loss_sum / len(train_loader), directory=model_directory)
        save_check_point(epoch, period=10)

    return


def train_pixel_based_network():
    for epoch in range(epoch_start, epoch_end, 1):
        loss_sum = 0.0

        for low_res, high_res in tqdm(train_loader, leave=False, desc='train'):
            # Load data: [-1 ~ 1]
            low_res = low_res.cuda()
            high_res = high_res.cuda()

            # Initialize gradient
            optimizer.zero_grad()

            # Prediction
            low_res = (low_res + 1.0) * 0.5                 # [-1 ~ 1] -> [ 0 ~ 1]
            pred_high_res = model(low_res)
            pred_high_res = (pred_high_res * 2.0) - 1.0     # [ 0 ~ 1] -> [-1 ~ 1]

            # Loss
            loss = criterion(pred_high_res, high_res)

            # Back-propagation
            loss.backward()
            optimizer.step()

            loss_sum = loss_sum + loss.detach().cpu()


            wandb.log({'Train/loss': loss.item()})

        # Logging
        print_log(epoch, loss_sum, loss_sum / len(train_loader), directory=model_directory)
        save_check_point(epoch, period=10)

    return


if __name__ == '__main__':
    # Parse the arguments
    parser = argparse.ArgumentParser(description="Train a LiDAR super-resolution network")
    parser.add_argument('-c', '--config',
                        type=str,
                        required=True,
                        help='Configuration filename. [.yaml]')
    parser.add_argument('-b', '--batch',
                        type=int,
                        required=False,
                        default=16,
                        help='Batch size for network training. (default: 16)')
    parser.add_argument('-cp', '--checkpoint',
                        type=str,
                        required=False,
                        default=None,
                        help='Check point filename. [.pth]')
    args = parser.parse_args()
    

    # print(torch.distributed.get_rank())
    # exit(0)

    # Log writer
   
    # Load the configurations
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    n_gpus = torch.cuda.device_count()

    # Train settings
    batch_size = args.batch
    train_dataset = generate_dataset(config['dataset'])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)

    model = generate_model(config['model']['name'], config['model']['args']).train()
    optimizer = optim.Adam(params=list(model.parameters()), lr=1e-4)
    lr_scheduler = MultiStepLR(optimizer, milestones=[200, 400, 600, 800], gamma=0.5)
    criterion = nn.L1Loss()
    epoch_start = 0
    epoch_end = 1000
    # epoch_end = 1

    # Load a valid check point
    check_point = torch.load(args.checkpoint) if args.checkpoint is not None else None
    if check_point is not None:
        if is_valid_check_point():
            model.load_state_dict(check_point['model']['state_dict'])
            epoch_start = check_point['epoch']
            optimizer.load_state_dict(check_point['optimizer'])
            lr_scheduler.load_state_dict(check_point['lr_scheduler'])
        else:
            print('ERROR: Invalid check point file:', args.checkpoint)
            exit(0)

    # Set the multi-gpu for training
    if n_gpus > 1:
        model = nn.parallel.DataParallel(model)

    if config['logger']['wandb_disabled']:
        mode = "disabled"
    else:
        mode = "online"
    wandb.init(project=config['logger']['project_name'],
                entity=config['logger']['entity'],
                name = config['logger']['run_name'], 
                mode=mode,
                group=config['logger']['group_name'],)

    model.cuda()

    print("=================== Training Configuration ====================  ")
    model_name = config['model']['name']
    model_directory = config['model']['output']
    print('  Model:', model_name, '(' + str(sum(p.numel() for p in model.parameters() if p.requires_grad)) + ' parameters)')
    for key, value in config['model']['args'].items():
        print('    ' + key + ':', value)
    print('  Output directory:', model_directory)
    print('  Check point file:', args.checkpoint)
    print('  ')
    print('  Dataset:', config['dataset']['name'], '[' + config['dataset']['type'] + '] (' + str(len(train_dataset)) + ' pairs)')
    print('  Batch:', batch_size)
    print('  Epoch:', epoch_start, '-->', epoch_end)
    print('  GPUs:', n_gpus)
    print("===============================================================  \n")

    # NOTE: The training approaches are different according to the type of network structure
    if config['dataset']['type'] in ['range_images', 'range_images_durlar', 'range_images_kitti']:
        train_pixel_based_network()
    elif config['dataset']['type'] in ['range_samples_from_image', 'range_samples_from_durlar', 'range_samples_from_kitti']:
        train_implicit_network()



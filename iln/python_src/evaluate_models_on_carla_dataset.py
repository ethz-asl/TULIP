import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Datasets
from dataset.samples_from_image_dataset import SamplesFromImageDataset
from dataset.dataset_utils import generate_dataset, denormalization_ranges

# Models
from models.iln.iln import ILN
from models.liif_cvpr21.liif_lidar import LIIFLiDAR
from models.interpolation.interpolation import Interpolation
from models.lsr_ras20.unet import UNet
from models.model_utils import generate_model

# Metric
from metric.mae_evaluator import MAEEvaluator
from metric.voxel_iou_evaluator import VoxelIoUEvaluator
import yaml


def test_pixel_based_network():
    mae_evaluator = MAEEvaluator()
    voxel_evaluator = VoxelIoUEvaluator(voxel_size=args.voxel_size, lidar=test_dataset.lidar_out)

    for packed_batches in tqdm(test_loader, leave=False, desc='test'):
        # input_range_image:    [N, 1, H_in, W_in]
        # output_ranges:        [N, H_out*W_out, 1]
        input_range_image, output_ranges = packed_batches[0].cuda(), packed_batches[2].cuda()

        # Prediction: input_range_image [N, 1, H_in, W_in] --> pred_ranges: [N, H_out*W_out, 1]
        with torch.no_grad():
            # [-1 ~ 1] -> [ 0 ~ 1]
            input_range_image += 1.0
            input_range_image *= 0.5
            pred_ranges = model(input_range_image).view(output_ranges.shape[0], -1, 1)
            # [ 0 ~ 1] -> [-1 ~ 1]
            pred_ranges *= 2.0
            pred_ranges -= 1.0

        # Evaluations
        output_ranges = denormalization_ranges(output_ranges)  # [N * H_out * W_out]
        pred_ranges = denormalization_ranges(pred_ranges)  # [N * H_out * W_out]
        pred_ranges[pred_ranges < 0.] = 0.
        pred_ranges[pred_ranges > test_dataset.lidar_out['norm_r']] = test_dataset.lidar_out['norm_r']

        # MAE
        mae_evaluator.update((output_ranges.flatten(), pred_ranges.flatten()))

        # Voxel IoU
        output_range_images = output_ranges.view(-1, test_dataset.lidar_out['channels'] * test_dataset.lidar_out['points_per_ring'])
        pred_range_images = pred_ranges.view(-1, test_dataset.lidar_out['channels'] * test_dataset.lidar_out['points_per_ring'])
        voxel_evaluator.update(pred_range_images.detach().cpu().numpy(), output_range_images.detach().cpu().numpy())

    return mae_evaluator.compute(), voxel_evaluator.compute()


def test_implicit_network(pred_batch=1):
    mae_evaluator = MAEEvaluator()
    voxel_evaluator = VoxelIoUEvaluator(voxel_size=args.voxel_size, lidar=test_dataset.lidar_out)

    for packed_batches in tqdm(test_loader, leave=False, desc='test'):
        # input_range_image:    [N, 1, H_in, W_in]
        # input_queries:        [N, H_out*W_out, 2]
        # output_ranges:        [N, H_out*W_out, 1]
        input_range_image, input_queries, output_ranges = packed_batches[0].cuda(), packed_batches[1].cuda(), packed_batches[2].cuda()

        # Prediction: input_range_image [N, 1, H_in, W_in] --> pred_ranges: [N, H_out*W_out, 1]
        if input_range_image.shape[0] == 1 and pred_batch > 1:
            with torch.no_grad():
                input_range_image = input_range_image.repeat(num_of_gpus, 1, 1, 1)   # for multi-gpus
                input_queries = input_queries.view(num_of_gpus * pred_batch, -1, 2)  #
                preds = []
                for n in range(pred_batch):
                    bs = n * num_of_gpus
                    be = (n + 1) * num_of_gpus
                    preds.append(model(input_range_image, input_queries[bs:be, :, :]).view(1, -1, 1))
                pred_ranges = torch.cat(preds, dim=1).view(1, -1, 1)
        else:
            with torch.no_grad():
                pred_ranges = model(input_range_image, input_queries)

        # Evaluations
        output_ranges = denormalization_ranges(output_ranges)   # [N * H_out * W_out]
        pred_ranges = denormalization_ranges(pred_ranges)       # [N * H_out * W_out]
        pred_ranges[pred_ranges < 0.] = 0.
        pred_ranges[pred_ranges > test_dataset.lidar_out['norm_r']] = test_dataset.lidar_out['norm_r']



        # MAE
        mae_evaluator.update((output_ranges.flatten(), pred_ranges.flatten()))

        # Voxel IoU
        output_range_images = output_ranges.view(-1, test_dataset.lidar_out['channels'] * test_dataset.lidar_out['points_per_ring'])
        pred_range_images = pred_ranges.view(-1, test_dataset.lidar_out['channels'] * test_dataset.lidar_out['points_per_ring'])
        voxel_evaluator.update(pred_range_images.detach().cpu().numpy(), output_range_images.detach().cpu().numpy())

    return mae_evaluator.compute(), voxel_evaluator.compute()


if __name__ == '__main__':
    # Parse the arguments
    parser = argparse.ArgumentParser(description="Evaluate the MAE (Mean Absolute Error) on the Carla dataset")
    # parser.add_argument('-d', '--carla_directory',
    #                     type=str,
    #                     required=True,
    #                     help='Carla dataset directory')
    parser.add_argument('-c', '--config',
                        type=str,
                        required=True,
                        help='Configuration filename. [.yaml]')
    parser.add_argument('-o', '--output_directory',
                        type=str,
                        required=False,
                        default=None,
                        help='Output directory')
    parser.add_argument('-b', '--batch',
                        type=int,
                        required=False,
                        default=7,
                        choices=[1, 2, 7, 11, 14, 17, 22, 34],
                        help='Batch size for network testing. (default: 7)')
    parser.add_argument('-cp', '--checkpoint',
                        type=str,
                        required=True,
                        help='Check point filename. [.pth]')
    # parser.add_argument('-r', '--target_resolution',
    #                     type=str,
    #                     required=True,
    #                     choices=['64_1024', '128_2048', '256_4096'],
    #                     help='Target resolution')
    parser.add_argument('-v', '--voxel_size',
                        type=float,
                        required=False,
                        default=0.1,
                        help='Voxel size for visualization, Default: 0.1')
    args = parser.parse_args()

    # Load the check point
    check_point = torch.load(args.checkpoint)

    # Test settings
    dataset_config = {'name': 'Carla',
                      'type': 'range_samples_from_image',
                      'args': {'directory': args.carla_directory,
                               'scene_ids': ['Town07', 'Town10HD'],
                               'res_in': '16_1024',
                               'res_out': args.target_resolution}}



    # Load the configurations
    with open(args.config, 'r') as f:
        dataset_config = yaml.load(f, Loader=yaml.FullLoader)

    # Generate dataset
    batch_size = args.batch
    test_dataset = generate_dataset(dataset_config)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    # Model
    model = generate_model(check_point['model']['name'], check_point['model']['args'])
    model.load_state_dict(check_point['model']['state_dict'])
    num_of_gpus = torch.cuda.device_count()
    if torch.cuda.device_count() > 1:
        model = nn.parallel.DataParallel(model)
    model.eval().cuda()

    # Output directory
    output_directory = args.output_directory if args.output_directory is not None else os.path.dirname(args.checkpoint)
    eval_result_filename = os.path.splitext(os.path.basename(args.checkpoint))[0]
    eval_result_filename = os.path.join(output_directory, eval_result_filename + '[' + dataset_config['args']['res_out'] + ']')

    print("======================= Configuration ========================  ")
    model_name = check_point['model']['name']
    model_directory = args.checkpoint
    print('  Model:', model_name,
          '(' + str(sum(p.numel() for p in model.parameters() if p.requires_grad)) + ' parameters)')
    for key, value in check_point['model']['args'].items():
        print('    ' + key + ':', value)
    print('  Check point file:', args.checkpoint)
    print('  ')
    print('  Dataset: Carla', '(' + str(len(test_dataset)) + ' pairs)')
    print('  Batch:', batch_size)
    print('  Output filename:', eval_result_filename)
    print('  Target resolution:', test_dataset.lidar_out['channels'], 'x', test_dataset.lidar_out['points_per_ring'])
    print("==============================================================  \n")

    # Evaluate the network
    if check_point['model']['name'].find('lsr') != -1:
        mae, voxel_ious = test_pixel_based_network()
    else:
        mae, voxel_ious = test_implicit_network()

    # Report the evaluation result
    if not os.path.isfile(eval_result_filename):
        eval_result_file = open(eval_result_filename, 'w')
        eval_result_file.write('epoch |    mae   |    iou   |    pre   |    rec   |    f1    \n')   # HEADER
        eval_result_file.close()

    eval_result_file = open(eval_result_filename, 'a')
    eval_result_file.write('%d\t%f   %f   %f   %f   %f\n' % (check_point['epoch'], mae, voxel_ious[0], voxel_ious[1], voxel_ious[2], voxel_ious[3]))
    eval_result_file.close()

    print('Evaluate the model:', model_directory)

import argparse
import os
import copy

import torch
import torch.nn as nn

# Datasets
from dataset.dataset_utils import read_range_image_binary, write_range_image_binary, generate_laser_directions
from dataset.dataset_utils import normalization_ranges, denormalization_ranges, normalization_queries

# Models
from models.iln.iln import ILN
from models.liif_cvpr21.liif_lidar import LIIFLiDAR
from models.interpolation.interpolation import Interpolation
from models.model_utils import generate_model

# Other utils
from visualization.visualization_utils import draw_range_image


def predict_detection_distances(input_image, pixel_centers, pred_batch=100000):
    """
    Predict a high-resolution range image (prediction) from low-resolution image (input) and pixel centers (queries).

    :param input_image: low-resolution LiDAR range image
    :param pixel_centers: query lasers associated with pixel centers of range image
    :param pred_batch: batch size for predicting the detection distances (default: 100000)
    :return high-resolution LiDAR range image
    """
    input_image = torch.from_numpy(input_image)[None, None, :, :].cuda()
    pixel_centers = torch.from_numpy(pixel_centers)[None, :, :].cuda()

    with torch.no_grad():
        model.gen_feat(input_image)
        n = pixel_centers.shape[1]
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + pred_batch, n)
            pred = model.query_detection(pixel_centers[:, ql:qr, :])
            preds.append(pred)
            ql = qr

    return torch.cat(preds, dim=1).view(-1).cpu().numpy()


if __name__ == '__main__':
    # Parse the arguments
    parser = argparse.ArgumentParser(description="Demonstrate the LiDAR image super-resolution to a target resolution")
    parser.add_argument('-i', '--input_filename',
                        type=str,
                        required=True,
                        help='Input range image [.rimg]; e.g.) Carla/Town07/16_1024/1.rimg')
    parser.add_argument('-cp', '--checkpoint',
                        type=str,
                        required=True,
                        help='Check point filename. [.pth]')
    parser.add_argument('-od', '--output_directory',
                        type=str,
                        required=False,
                        default=None,
                        help='Directory to save the reconstructed range image (default: check point\'s directory)')
    parser.add_argument('-r', '--target_resolution',
                        type=str,
                        required=False,
                        default='128_2048',
                        help='Vertical and horizontal target resolution; (default: 128_2048)')
    args = parser.parse_args()

    # Load the check point
    check_point = torch.load(args.checkpoint)

    # Model
    model = generate_model(check_point['model']['name'], check_point['model']['args'])
    model.load_state_dict(check_point['model']['state_dict'])
    if torch.cuda.device_count() > 1:
        model = nn.parallel.DataParallel(model)
    model.eval().cuda()

    # Prepare the trained LiDAR specification
    lidar_in = check_point['lidar_in']
    lidar_out = copy.deepcopy(check_point['lidar_in'])
    lidar_out['channels'] = int(args.target_resolution.split('_')[0])
    lidar_out['points_per_ring'] = int(args.target_resolution.split('_')[1])

    print("===================== Demo Configuration ======================  ")
    check_point_filename = args.checkpoint
    model_name = check_point['model']['name']
    input_filename = args.input_filename
    output_directory = args.output_directory if args.output_directory is not None else os.path.dirname(check_point_filename)
    print('  Check point file:', check_point_filename)
    print('  Model:', model_name, '(' + str(sum(p.numel() for p in model.parameters() if p.requires_grad)) + ' parameters)')
    for key, value in check_point['model']['args'].items():
        print('    ' + key + ':', value)
    print('  ')
    print('  Input filename:', args.input_filename)
    print('  Input resolution:', lidar_in['channels'], 'x', lidar_in['points_per_ring'])
    print('  Output resolution:', lidar_out['channels'], 'x', lidar_out['points_per_ring'])
    print('  Output directory:', output_directory)
    print("===============================================================  \n")

    # Read the input range image (normalized)
    input_range_image = read_range_image_binary(input_filename, lidar=lidar_in)
    input_range_image = normalization_ranges(input_range_image, norm_r=lidar_in['norm_r'])

    # Generate the query lasers (normalized)
    query_lasers = generate_laser_directions(lidar_out)
    query_lasers = normalization_queries(query_lasers, lidar_in)

    # Reconstruct the up-scaled output range image (normalized)
    pred_detection_distances = predict_detection_distances(input_image=input_range_image,
                                                           pixel_centers=query_lasers,
                                                           pred_batch=100000)
    pred_range_image = pred_detection_distances.reshape(lidar_out['channels'], lidar_out['points_per_ring'])

    # 1. Draw the input range image
    output_filename = os.path.join(output_directory, 'demo_input_range_image.png') if output_directory else None
    draw_range_image(range_image=input_range_image, filename=output_filename)
    print('Save the input range image:', output_filename)

    # 2. Draw the output range image
    output_filename = os.path.join(output_directory, 'demo_output_range_image.png') if output_directory else None
    draw_range_image(range_image=pred_range_image, filename=output_filename)
    print('Save the output range image:', output_filename)

    # 3. Save the output range image [.rimg]
    output_filename = os.path.join(output_directory, 'output_range_image.rimg') if output_directory else None
    denormalized_pred_range_image = denormalization_ranges(pred_range_image, norm_r=lidar_out['norm_r'])
    write_range_image_binary(range_image=denormalized_pred_range_image, filename=output_filename)
    print('Save the output range image [.rimg]:', output_filename)


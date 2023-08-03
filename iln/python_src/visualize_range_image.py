import argparse

from dataset.dataset_utils import read_range_image_binary, normalization_ranges
from visualization.visualization_utils import draw_range_image


if __name__ == '__main__':
    # Parse the arguments
    parser = argparse.ArgumentParser(description="Visualize a range image")
    parser.add_argument('-i', '--input_filename',
                        type=str,
                        required=True,
                        help='Input range image [.rimg]; e.g.) Carla/Town07/16_1024/1.rimg')
    parser.add_argument('-o', '--output_filename',
                        type=str,
                        required=False,
                        default=None,
                        help='Output range image [.png, .jpg]')
    parser.add_argument('-m', '--mask',
                        action='store_true',
                        help='Enable the invalid mask for visualization')
    parser.add_argument('-vl', '--min_value',
                        type=float,
                        required=False,
                        default=2.0,
                        help='Minimum (lowest) value of detection range')
    parser.add_argument('-vh', '--max_value',
                        type=float,
                        required=False,
                        default=80.0,
                        help='Maximum (highest) value of detection range')
    parser.add_argument('-vn', '--norm_value',
                        type=float,
                        required=False,
                        default=100.0,
                        help='Normalization parameter')
    args = parser.parse_args()

    print("  ================== Range Image Visualization ==================  ")
    min_value = args.min_value
    max_value = args.max_value
    norm_value = args.norm_value
    input_filename = args.input_filename
    output_filename = args.output_filename
    use_mask = args.mask
    print('  Input filename:', input_filename)
    print('  Output filename:', output_filename)
    print('  Minimum range:', min_value, '[m]')
    print('  Maximum range:', max_value, '[m]')
    print('  Normalization value:', norm_value)
    print('  Use mask:', use_mask)
    print("  ===============================================================  \n")

    # Read a range image (denormalized)
    range_image = read_range_image_binary(input_filename)
    range_image[range_image < min_value] = norm_value
    range_image[range_image > max_value] = norm_value
    valid_mask = (range_image != norm_value) if use_mask else None

    # Normalize the range image: [0 ~ norm_r] --> [0 ~ 1] --> [-1 ~ 1]
    range_image = normalization_ranges(range_image, norm_r=norm_value)
    print('Image size:', range_image.shape)

    # Draw the range image (or save the range image if an output filename is given)
    draw_range_image(range_image=range_image, filename=output_filename, vis_mask=valid_mask)

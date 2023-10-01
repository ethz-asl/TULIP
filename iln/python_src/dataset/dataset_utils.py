import numpy as np
import yaml
from PIL import Image


dataset_list = {}


def register_dataset(name):
    def decorator(cls):
        dataset_list[name] = cls
        return cls
    return decorator


def generate_dataset(dataset_config):
    """
    Generate a dataset depending on the dataset specification (configuration file [.yaml]).

    :param dataset_config: dataset specification
    :return: dataset
    """
    return dataset_list[dataset_config['type']](**dataset_config['args'])

def read_range_kitti(filename):
    range_intensity_map = np.load(filename)
    range_map = range_intensity_map[..., 0]

    return range_map.astype(np.float32)

def read_range_durlar(filename, dtype=np.float16, lidar=None):
    """
    Read a range image from a binary file.

    :param filename: filename of range image
    :param dtype: encoding type of binary data (default: float16)
    :param lidar: LiDAR specification for crop the invalid detection distances
    :return: range image encoded by float32 type
    """
    range_image_file = open(filename, 'rb')
    # range_image = Image.open(range_image_file).convert("L")
    range_image = np.load(range_image_file)
    # range_image = np.asarray(range_image, dtype = np.float32) / 255

    return range_image.astype(np.float32)


def downsample_range_durlar(range_image, h_high_res = 128, downsample_factor = 4):
    low_res_index = range(0, h_high_res, downsample_factor)
    return range_image[low_res_index, :]

def read_and_downsample_range_image_binary(filename, dtype=np.float16, lidar=None, downsample_factor = 4):
    """
    Read a range image from a binary file.

    :param filename: filename of range image
    :param dtype: encoding type of binary data (default: float16)
    :param lidar: LiDAR specification for crop the invalid detection distances
    :return: range image encoded by float32 type
    """
    range_image_file = open(filename, 'rb')

    # Read the size of range image
    size = np.fromfile(range_image_file, dtype=np.uint, count=2)

    # Read the range image
    range_image = np.fromfile(range_image_file, dtype=dtype)
    range_image = range_image.reshape(size[1], size[0])
    range_image = range_image.transpose()
    range_image = range_image.astype(np.float32)


    low_res_index = range(0, size[0], downsample_factor)
    range_image = range_image[low_res_index, :]

    if lidar is not None:
        # Crop the values out of the detection range
        range_image[range_image < 10e-10] = lidar['norm_r']
        range_image[range_image < lidar['min_r']] = 0.0
        range_image[range_image > lidar['max_r']] = lidar['norm_r']

    range_image_file.close()

    return range_image.astype(np.float32)



def read_range_image_binary(filename, dtype=np.float16, lidar=None):
    """
    Read a range image from a binary file.

    :param filename: filename of range image
    :param dtype: encoding type of binary data (default: float16)
    :param lidar: LiDAR specification for crop the invalid detection distances
    :return: range image encoded by float32 type
    """
    range_image_file = open(filename, 'rb')

    # Read the size of range image
    size = np.fromfile(range_image_file, dtype=np.uint, count=2)

    # Read the range image
    range_image = np.fromfile(range_image_file, dtype=dtype)
    range_image = range_image.reshape(size[1], size[0])
    range_image = range_image.transpose()
    range_image = range_image.astype(np.float32)

    if lidar is not None:
        # Crop the values out of the detection range
        range_image[range_image < 10e-10] = lidar['norm_r']
        range_image[range_image < lidar['min_r']] = 0.0
        range_image[range_image > lidar['max_r']] = lidar['norm_r']

    range_image_file.close()

    return range_image.astype(np.float32)


def write_range_image_binary(filename, range_image, dtype=np.float16):
    """
    Write a range image to a binary file.

    :param filename: filename of range image
    :param range_image: range image
    :param dtype: encoding type of binary data (default: float16)
    """
    range_image_file = open(filename, 'wb')
    range_image = range_image.transpose().astype(dtype)

    # Write the size of range image
    size = [range_image.shape[1], range_image.shape[0]]
    size = np.array(size, dtype=np.uint)
    size.tofile(range_image_file)

    # Write the range image
    range_image.tofile(range_image_file)

    range_image_file.close()

    return range_image


def read_range_samples_binary(filename, dtype=np.float16):
    """
    Read range samples from a binary file.

    :param filename: filename of range samples
    :param dtype: encoding type of binary data (default: float16)
    :return: range samples encoded by float32 type
    """
    range_samples_file = open(filename, 'rb')

    # Read the size of point array
    num_of_samples = np.fromfile(range_samples_file, dtype=np.uint, count=1)[0]

    # Read the range samples
    samples = np.fromfile(range_samples_file, dtype=dtype)
    samples = samples.reshape(num_of_samples, 3)

    return samples.astype(np.float32)


def write_range_samples_binary(filename, range_samples, dtype=np.float16):
    """
    Write range samples to a binary file.

    :param filename: filename of range samples
    :param range_samples: range samples (dtype: float32)
    :param dtype: encoding type of binary data
    """
    range_samples_file = open(filename, 'wb')
    range_samples = range_samples.astype(dtype)

    # Write the size of point array
    num_of_samples = [range_samples.shape[0]]
    num_of_samples = np.array(num_of_samples, dtype=np.uint)
    num_of_samples.tofile(range_samples_file)

    # Write the range samples
    range_samples.tofile(range_samples_file)

    range_samples_file.close()

    return


def initialize_lidar(filename, channels, points_per_ring):
    """
    Initialize a LiDAR having given laser resolutions from a configuration file.

    :param filename: LiDAR configuration filename [.yaml]
    :param channels: number of vertical angles (vertical resolution)
    :param points_per_ring: number of horizontal angles (horizontal resolution)
    :return: LiDAR specification
    """
    with open(filename, 'r') as f:
        lidar = yaml.load(f, Loader=yaml.FullLoader)

    lidar['channels'] = channels
    lidar['points_per_ring'] = points_per_ring
    lidar['max_v'] *= (np.pi / 180.0)  # [rad]
    lidar['min_v'] *= (np.pi / 180.0)  # [rad]
    lidar['max_h'] *= (np.pi / 180.0)  # [rad]
    lidar['min_h'] *= (np.pi / 180.0)  # [rad]

    return lidar


def generate_laser_directions(lidar):
    """
    Generate the laser directions using the LiDAR specification.

    :param lidar: LiDAR specification
    :return: a set of the query laser directions;
    """
    v_dir = np.linspace(start=lidar['min_v'], stop=lidar['max_v'], num=lidar['channels'])
    h_dir = np.linspace(start=lidar['min_h'], stop=lidar['max_h'], num=lidar['points_per_ring'], endpoint=False)

    v_angles = []
    h_angles = []

    for i in range(lidar['channels']):
        v_angles = np.append(v_angles, np.ones(lidar['points_per_ring']) * v_dir[i])
        h_angles = np.append(h_angles, h_dir)

    return np.stack((v_angles, h_angles), axis=-1).astype(np.float32)


def range_image_to_points(range_image, lidar, remove_zero_range=True):
    """
    Convert a range image to the points in the sensor coordinate.

    :param range_image: denormalized range image
    :param lidar: LiDAR specification
    :param remove_zero_range: flag to remove the points with zero ranges
    :return: points in sensor coordinate
    """
    angles = generate_laser_directions(lidar)
    r = range_image.flatten()

    x = np.sin(angles[:, 1]) * np.cos(angles[:, 0]) * r
    y = np.cos(angles[:, 1]) * np.cos(angles[:, 0]) * r
    z = np.sin(angles[:, 0]) * r

    points = np.stack((x, y, z), axis=-1)  # sensor coordinate

    # Remove the points having invalid detection distances
    if remove_zero_range is True:
        points = np.delete(points, np.where(r < 1e-5), axis=0)

    return points


def points_to_ranges(points):
    """
    Convert points in the sensor coordinate into the range data in spherical coordinate.

    :param points: points in sensor coordinate
    :return: the range data in spherical coordinate
    """
    # sensor coordinate
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    r = np.sqrt(x * x + y * y + z * z)
    v = np.arctan2(z, np.sqrt(x * x + y * y))
    h = np.arctan2(x, y)

    return np.stack((v, h, r), axis=-1)


def points_to_range_image(points, lidar):
    """
    Convert points in the sensor coordinate to a range image.

    :param points: points in sensor coordinate
    :param lidar: LiDAR specification
    :return: range image of which a value has the range [0 ~ norm_r]
    """
    range_samples = points_to_ranges(points)

    range_image = np.zeros([lidar['channels'], lidar['points_per_ring']], dtype=np.float32)
    max_y = max(lidar['max_v'], np.max(range_samples[:, 0]))
    min_y = min(lidar['min_v'], np.min(range_samples[:, 0]))
    res_y = (max_y - min_y) / (lidar['channels']-1)  # include the last
    res_x = (lidar['max_h'] - lidar['min_h']) / lidar['points_per_ring']              # exclude the last

    # offset to match a point into a pixel center
    range_samples[:, 0] += (res_y * 0.5)
    range_samples[:, 1] += (res_x * 0.5)
    # horizontal values are within [-pi, pi)
    range_samples[range_samples[:, 1] < -np.pi, 1] += (2.0 * np.pi)
    range_samples[range_samples[:, 1] >= np.pi, 1] -= (2.0 * np.pi)
    # Pixel Index --> Ideally one pixel is assigned to one sample
    py = np.trunc((range_samples[:, 0] - min_y) / res_y).astype(np.int)
    # py = np.trunc((range_samples[:, 0] - lidar_spec['min_v']) / res_y).astype(np.int)
    px = np.trunc((range_samples[:, 1] - lidar['min_h']) / res_x).astype(np.int)

    # Insert the ranges
    range_image[py, px] = range_samples[:, 2]

    # Crop the values out of the detection range
    range_image[range_image < 10e-10] = lidar['norm_r']
    range_image[range_image < lidar['min_r']] = 0.0
    range_image[range_image > lidar['max_r']] = lidar['norm_r']

    return range_image


def normalization_ranges(range_image, norm_r=100.0):
    """
    Normalize a range image: [0 ~ norm_r] --> [-1 ~ 1].

    :param range_image: a value has the range [0 ~ norm_r]
    :param norm_r: the maximum value of detection distance [m]
    :return: normalized range image of which each value has the range [-1, 1]
    """
    range_image *= (2.0 / norm_r)
    range_image -= 1.0
    return range_image


def denormalization_ranges(range_image, norm_r=120.0):
    """
    Denormalize a range image: [-1 ~ 1] --> [0 ~ norm_r].

    :param range_image: a value has the range [-1 ~ 1]
    :param norm_r: the maximum value of detection distance [m]
    :return: denormalized range image of which each value has the range [0 ~ norm_r]
    """
    range_image += 1.0
    # Needed for carla dataset
    # if norm_r > 1:
    range_image *= (0.5 * norm_r)
    return range_image


def normalization_queries(queries, lidar_in):
    """
    Normalize query lasers toward input range image space.
    [min_v-v_res*0.5 ~ max_v+v_res*0.5] --> [-1 ~ 1]
    [min_h-h_res*0.5 ~ max_h-h_res*0.5] --> [-1 ~ 1]

    :param queries: query lasers without normalization
    :param lidar_in: input LiDAR specification
    :return: normalized query lasers
    """
    # Vertical angle: [min_v-v_res*0.5 ~ max_v+v_res*0.5] --> [0 ~ 1]
    v_res = (lidar_in['max_v'] - lidar_in['min_v']) / (lidar_in['channels'] - 1)
    min_v = lidar_in['min_v'] - v_res * 0.5
    max_v = lidar_in['max_v'] + v_res * 0.5
    queries[:, 0] -= min_v
    queries[:, 0] /= (max_v - min_v)

    # Horizontal angle: [min_h-h_res*0.5 ~ max_h-h_res*0.5] --> [0 ~ 1]
    h_res = (lidar_in['max_h'] - lidar_in['min_h']) / lidar_in['points_per_ring']
    queries[:, 1] += (h_res * 0.5)
    queries[queries[:, 1] < -np.pi, 1] += (2.0 * np.pi)  # min_h == -np.pi
    queries[queries[:, 1] >= np.pi, 1] -= (2.0 * np.pi)  # max_h == +np.pi
    queries[:, 1] += np.pi
    queries[:, 1] /= (2.0 * np.pi)

    # [0 ~ 1] --> [-1 ~ 1]
    queries *= 2.0
    queries -= 1.0
    return queries


def denormalization_queries(queries, lidar_in):
    """
    Denormalize the query lasers from input range image space.
    [-1 ~ 1] --> [min_v-v_res*0.5 ~ max_v+v_res*0.5]
    [-1 ~ 1] --> [min_h-h_res*0.5 ~ max_h-h_res*0.5]

    :param queries: normalized query lasers
    :param lidar_in: input LiDAR specification
    :return: denormalized query lasers
    """
    # [-1 ~ 1] --> [0 ~ 1]
    queries += 1.0
    queries *= 0.5

    # Vertical angle: [0 ~ 1] --> [min_v-v_res*0.5 ~ max_v+v_res*0.5]
    v_res = (lidar_in['max_v'] - lidar_in['min_v']) / (lidar_in['channels'] - 1)
    min_v = lidar_in['min_v'] - v_res * 0.5
    max_v = lidar_in['max_v'] + v_res * 0.5
    queries[:, 0] *= (max_v - min_v)
    queries[:, 0] += min_v

    # Horizontal angle: [0 ~ 1] --> [min_h-h_res*0.5 ~ max_h-h_res*0.5]
    h_res = (lidar_in['max_h'] - lidar_in['min_h']) / lidar_in['points_per_ring']
    queries[:, 1] *= (2.0 * np.pi)
    queries[:, 1] += (np.pi - h_res * 0.5)
    queries[queries[:, 1] < -np.pi, 1] += (2.0 * np.pi)  # min_h == -np.pi
    queries[queries[:, 1] >= np.pi, 1] -= (2.0 * np.pi)  # max_h == +np.pi

    return queries

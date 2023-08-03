import rospy
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header

import matplotlib.pyplot as plt
import numpy as np


def get_pointcloud_xyz(points, stamp=None, frame_id='map'):
    """
    Get a ROS message of point cloud data.

    :param points: point clouds
    :param stamp: visualization time stamp
    :param frame_id: visualization frame id (default: 'map')
    :return: point cloud message (PointCloud2)
    """
    assert(points.shape[1] == 3)

    header = Header()
    header.frame_id = frame_id
    header.stamp = rospy.Time.now() if stamp is None else stamp

    fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)
    ]

    return pc2.create_cloud(header, fields, points)


def get_pointcloud_xyzi(points, stamp=None, frame_id='map'):
    """
    Get a ROS message of point cloud data with intensity labels.

    :param points: point cloud with intensity labels
    :param stamp: visualization time stamp
    :param frame_id: visualization frame id (default: 'map')
    :return: point cloud message (PointCloud2)
    """
    assert(points.shape[1] == 4)

    header = Header()
    header.frame_id = frame_id
    header.stamp = rospy.Time.now() if stamp is None else stamp

    fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1)
    ]

    return pc2.create_cloud(header, fields, points)


def draw_range_image(range_image, colormap='rainbow', filename=None, vis_mask=None):
    """
    Visualize the range image to a window or a file.

    :param range_image: normalized range image [-1 ~ 1]
    :param colormap: color map ID used in Matplotlib
    :param filename: output filename of range image
    :param vis_mask: boolean mask for visualization
    """
    plt.figure()

    image = np.flip(range_image, axis=0)

    if filename is None:
        valid = None if vis_mask is None else np.flip(vis_mask, axis=0).astype(float)
        plt.imshow(X=image, alpha=valid, cmap=colormap, vmin=-1.0, vmax=1.0, interpolation='none')
        plt.show()
    else:
        if vis_mask is None:
            plt.imsave(filename, arr=image, cmap=colormap, vmin=-1.0, vmax=1.0)
        else:
            cmap = plt.cm.get_cmap(colormap)
            norm = plt.Normalize(vmin=-1.0, vmax=1.0)
            image = cmap(norm(image))                       # mapped color image
            image[~(np.flip(vis_mask, axis=0)), 0:3] = 1.0  # white color
            plt.imsave(filename, arr=image)

    return

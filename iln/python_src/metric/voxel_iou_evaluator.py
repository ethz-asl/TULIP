import numpy as np
import math
import voxelizer


class VoxelIoUEvaluator:
    def __init__(self, voxel_size, lidar):
        self.voxel_size = voxel_size
        self.lidar = lidar
        self.iou_sum = 0.0
        self.precision_sum = 0.0
        self.recall_sum = 0.0
        self.f1_sum = 0.0
        self.cnt = 0

    def reset(self):
        self.iou_sum = 0.0
        self.precision_sum = 0.0
        self.recall_sum = 0.0
        self.f1_sum = 0.0
        self.cnt = 0

    def update(self, pred_rimg, gt_rimg):
        """
        Accumulate the errors between prediction and ground-truth point clouds in sensor's coordinate.

        :param pred_rimg: prediction range images
        :param gt_rimg: ground truth range images
        """
        results = voxelizer.compute_voxel_iou_of_images(gt_rimg, pred_rimg, self.voxel_size, self.lidar)
        self.iou_sum += np.sum(results[:, 0])
        self.precision_sum += np.sum(results[:, 1])
        self.recall_sum += np.sum(results[:, 2])
        self.f1_sum += np.sum(results[:, 3])
        self.cnt += pred_rimg.shape[0]

    def compute(self):
        """
        Compute the mean values of IoU, Precision, Recall and F1 score over all the pairs.

        :return: performances [IoU, Precision, Recall, F1]
        """
        iou = float(self.iou_sum / self.cnt) if self.cnt > 0 else math.inf
        precision = float(self.precision_sum / self.cnt) if self.cnt > 0 else math.inf
        recall = float(self.recall_sum / self.cnt) if self.cnt > 0 else math.inf
        f1 = float(self.f1_sum / self.cnt) if self.cnt > 0 else math.inf
        return iou, precision, recall, f1

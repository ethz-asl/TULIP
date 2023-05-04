import itertools
import os

import h5py
import torch
import numpy as np
#import open3d as o3d

from utils.transforms import getTransform, getRandomTransform, transformPoints, transformPointsBatch

class CODDAggSnippet(torch.utils.data.Dataset):
    '''Dataset for a single Snippet that returns pairs of point clouds in their own CS and the relative transform from p0 -> p1'''

    def __init__(self, spath, frame_rate=1, maxDist=None, transform=None):
        super().__init__()

        f = h5py.File(spath, 'r')
        self.point_cloud = f['point_cloud']
        self.lidar_pose = f['lidar_pose']

        self.nframes = self.point_cloud.shape[0]
        self.ncars = self.point_cloud.shape[1]

        self.frame_rate = frame_rate
        self.maxDist = maxDist
        self.transform = transform

        self.idxs = self.createIdx()

    def createIdx(self):
        self.combinations = tuple(itertools.combinations(range(self.ncars), 2))
        idxs = []
        for frame_idx in range(0, self.nframes, self.frame_rate):
            for (c0, c1) in self.combinations:
                if self.maxDist == None:
                    idxs.append((frame_idx, c0, c1))
                    continue

                t0 = self.lidar_pose[frame_idx, c0, :3]
                t1 = self.lidar_pose[frame_idx, c1, :3]
                d = np.linalg.norm(t0-t1)
                if d > self.maxDist:
                    continue

                idxs.append((frame_idx, c0, c1))
        return idxs

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        frame_idx, v0, v1 = self.idxs[idx]

        #Get transforms (global CS)
        t0 = getTransform(*self.lidar_pose[frame_idx, v0].tolist())
        t1 = getTransform(*self.lidar_pose[frame_idx, v1].tolist())

        #Compute relative transform from p0 to p1
        t = np.linalg.inv(t1) @ t0
        t = t.astype(np.float32)

        #Get points in each vehicles' coordinate system (no transforms)
        p0, p1 = self.point_cloud[frame_idx, [v0,v1]].astype(np.float32)
        p = np.concatenate((p0[None], p1[None]), axis=0)

        if self.transform:
            p, t = self.transform(p, t)

        return p, t

class CODDAggDataset(torch.utils.data.ConcatDataset):
    '''Creates aggregated dataset by chaining all snippets for training/testing/validation sets'''

    def __init__(self, path, mode, frame_rate = 25, maxDist = 30, transform=None):
        #self.frame_rate = frame_rate # means we have 5 frames per snippet (125frames snippet/25 framerate = 5 frames)
        #self.maxDist = maxDist    # avoids vehicles without significant overlap

        if mode == 'train':
            maps = (2,3,4,5,6,7)
        elif mode == 'val':
            maps = (1,)
        elif mode == 'test':
            maps = (10,)
        else:
            raise NotImplementedError()

        #get paths 
        paths = (f'{path}/{p}' for p in os.listdir(path) if any([f'm{i}v' in p for i in maps]))

        #calls parent with iterable list of datasets (one CODDAggSnippet dataset for each snippet described in paths)
        super(CODDAggDataset, self).__init__([CODDAggSnippet(sp, frame_rate, maxDist, transform) for sp in paths])


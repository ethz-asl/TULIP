
from util.datasets import build_durlar_dataset, build_carla_upsampling_dataset, RandomRollRangeMap
import os
import numpy as np
import argparse
import cv2

# np.random.seed(0)

# a = RandomRollRangeMap()
# b = a.shift



args = argparse.ArgumentParser('MAE pre-training', add_help=False)

args.data_path = "/cluster/work/riner/users/biyang/dataset/Carla"
args.crop = False
args.in_chans = 1
args.log_transform = False

args.img_size_low_res = (16, 1024)
args.img_size_high_res = (128, 2048)

# dataset_train = build_durlar_dataset(is_train = True, args = args)
# dataset_val = build_durlar_dataset(is_train = False, args = args)

dataset = build_carla_upsampling_dataset(is_train = True, args = args)

input = dataset[0][0][0]
output = dataset[0][1][0]

img = output.numpy().transpose(1, 2, 0)
img_2 = input.numpy().transpose(1, 2, 0)

cv2.imwrite('carla.png', img*255)
cv2.imwrite('carla_low.png', img_2*255)


# for data in dataset_train:
#     a = data[0]

#     print(a.max(), a.min())

#     print((a>1).sum())


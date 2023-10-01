#!/bin/bash



# Training Set
cp /cluster/work/riner/users/biyang/dataset/DurLAR/DurLAR_20210716/depth/*[0][2-9][0-9][0-9][0-9].npy /cluster/work/riner/users/biyang/dataset/depth_intensity_large_new/train/depth/all/
cp /cluster/work/riner/users/biyang/dataset/DurLAR/DurLAR_20210716/depth/*[1-9][0-9][0-9][0-9][0-9].npy /cluster/work/riner/users/biyang/dataset/depth_intensity_large_new/train/depth/all/
cp /cluster/work/riner/users/biyang/dataset/DurLAR/DurLAR_20210716/intensity/*[0][2-9][0-9][0-9][0-9].npy /cluster/work/riner/users/biyang/dataset/depth_intensity_large_new/train/intensity/all/
cp /cluster/work/riner/users/biyang/dataset/DurLAR/DurLAR_20210716/intensity/*[1-9][0-9][0-9][0-9][0-9].npy /cluster/work/riner/users/biyang/dataset/depth_intensity_large_new/train/intensity/all/

# Validation Set
cp /cluster/work/riner/users/biyang/dataset/DurLAR/DurLAR_20210716/depth/*[0][1][0][0-9][0-9].npy /cluster/work/riner/users/biyang/dataset/depth_intensity_large_new/val/depth/all/
cp /cluster/work/riner/users/biyang/dataset/DurLAR/DurLAR_20210716/intensity/*[0][1][0][0-9][0-9].npy /cluster/work/riner/users/biyang/dataset/depth_intensity_large_new/val/intensity/all/

Hi there,
 
Thanks for your attention of DurLAR dataset (Li et al., 3DV 2021).
 
You will be able to download the DurLAR dataset using the command line (run in the Ubuntu Terminal). For the first time, it’s very likely that you need to make the durlar_download file executable, using follow command,

``` bash
chmod +x durlar_download
```
 
By default, this dataset downloads the small subset for simple testing. Use the following command: 

```bash
./durlar_download
```
 
At the same time, you can also choose to download datasets of different sizes and test drives. 

```
usage: ./durlar_download [dataset_sample_size] [drive]
dataset_sample_size = [ small | medium | full ]
drive = 1 ... 5
```
 
The DurLAR dataset is very large, so please download the full dataset only when necessary, and use the following command: 

```bash
./durlar_download full 5
```
 
Your network must not have any problems during the entire download process. In case of network problems, please delete all DurLAR dataset folder and re-run the download command.
 
The download script is now only support Ubuntu (tested on Ubuntu 18.04 and Ubuntu 20.04, amd64) for now. Please refer to https://collections.durham.ac.uk/collections/r2gq67jr192 to download the dataset for other OS manually.
 
If you are making use of this work in any way (including our dataset and toolkits), you must please reference the following paper in any report, publication, presentation, software release or any other associated materials:
 
DurLAR: A High-fidelity 128-channel LiDAR Dataset with Panoramic Ambient and Reflectivity Imagery for Multi-modal Autonomous Driving Applications (Li Li, Khalid N. Ismail, Hubert P. H. Shum and Toby P. Breckon), In Int. Conf. 3D Vision, 2021.
 
```bibtex
@inproceedings{li21durlar,
author = {Li, L. and Ismail, K.N. and Shum, H.P.H. and Breckon, T.P.},
title = {DurLAR: A High-fidelity 128-channel LiDAR Dataset with Panoramic Ambient and Reflectivity Imagery for Multi-modal Autonomous Driving Applications},
booktitle = {Proc. Int. Conf. on 3D Vision},
year = {2021},
month = {December},
publisher = {IEEE},
keywords = {autonomous driving, dataset, high resolution LiDAR, flash LiDAR, ground truth depth, dense depth, monocular depth estimation, stereo vision, 3D},
note = {to appear},
category = {automotive 3Dvision},
}
```
 
We will notify you by email of any updates to this dataset. If you encounter problems or bugs in use, please reply to this email directly or go to our GitHub repository at https://www.github.com/l1997i/DurLAR. 
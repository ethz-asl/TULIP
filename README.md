# TULIP: Transformer for Upsampling of LiDAR Point Clouds
This is an official implementation of the paper [TULIP: Transformer for Upsampling of LiDAR Point Clouds](https://arxiv.org/abs/2312.06733): A framework for LiDAR upsampling using Swin Transformer (accepted for publication at CVPR2024)
## Demo
The visualization is done by sampling a time-series subset from the test split
| KITTI                     |DurLAR                                                  |CARLA               |
| -------------------------------------------------------| ------------------------- | ------------------------------------------------------ |
| [![KITTI](http://img.youtube.com/vi/652crBsy6K4/0.jpg)](https://youtu.be/652crBsy6K4) | [![DurLAR](http://img.youtube.com/vi/c0fOlVC-I5Y/0.jpg)](https://youtu.be/c0fOlVC-I5Y)|[![CARLA](http://img.youtube.com/vi/gQ3jd9Z80vo/0.jpg)](https://youtu.be/gQ3jd9Z80vo)|

## Installation
Our work is implemented with the following environmental setups:
* Python == 3.8
* PyTorch == 1.12.0
* CUDA == 11.3

You can use conda to create the correct environment:
```
conda create -n myenv python=3.8
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
```

Then, install the dependencies in the environment:
```
pip install -r requirements.txt
pip install git+'https://github.com/otaheri/chamfer_distance'  # need access to gpu for compilation
```
You can refer to more details about chamfer distance package from https://github.com/otaheri/chamfer_distance

## Data Preparation
We have evaluated our method on three different datasets and they are all open source datasets:
* KITTI Raw Dataset: https://www.cvlibs.net/datasets/kitti/index.php 
* CARLA (collected from CARLA Simulator): https://github.com/PinocchioYS/iln (We use the same dataset as ILN)
* DurLAR: https://github.com/l1997i/DurLAR

After downloading the raw dataset, create train and test split for LiDAR upsampling:
```
bash bash_scripts/create_durlar_dataset.sh
bash bash_scripts/create_kitti_dataset.sh
```
The new dataset should be structured in this way:
```
dataset
│
└───KITTI / DurLAR
   │
   └───train
   │   │   00000001.npy
   │   │   00000002.npy
   │   │   ...
   └───val
       │   00000001.npy
       │   00000002.npy
       │   ...
```

## Training
We provide some bash files for running the experiment quickly with default settings. 
```
bash bash_scripts/tulip_upsampling_kitti.sh (KITTI)
bash bash_scripts/tulip_upsampling_carla.sh (CARLA)
bash bash_scripts/tulip_upsampling_durlar.sh (DurLAR)
```

## Evaluation
You can download the pretrained models from the [link](https://drive.google.com/file/d/15Ty7sKOrFHhB94vLBJOKasXaz1_DCa8o/view?usp=drive_link) and use them for evaluation.
```
bash bash_scripts/tulip_evaluation_kitti.sh (KITTI)
bash bash_scripts/tulip_evaluation_carla.sh (CARLA)
bash bash_scripts/tulip_evaluation_durlar.sh (DurLAR)
```

## Citation
```
@article{yang2023tulip,
  title={TULIP: Transformer for Upsampling of LiDAR Point Cloud},
  author={Yang, Bin and Pfreundschuh, Patrick and Siegwart, Roland and Hutter, Marco and Moghadam, Peyman and Patil, Vaishakh},
  journal={arXiv preprint arXiv:2312.06733},
  year={2023}
}
```

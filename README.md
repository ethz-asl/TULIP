# TULIP: Transformer for Upsampling of LiDAR Point Cloud
This is an official implementation of the paper [TULIP: Transformer for Upsampling of LiDAR Point Cloud](https://arxiv.org/abs/2312.06733): A framework for LiDAR upsampling using Swin Transformer.
## Demo
The visualization is done by sampling a time-series subset from the test split
| Method \ Dataset                                        | KITTI                     |DurLAR                                                  |CARLA               |
| -------------------------------------------------------| ------------------------- | ------------------------------------------------------ | ------                   |
| Low Resolution (Input)                                  | [![KITTI low resolution](http://img.youtube.com/vi/B42ZLbB1Qzs/0.jpg)](https://youtu.be/B42ZLbB1Qzs) | [![DurLAR low resolution](http://img.youtube.com/vi/Fu22sGp0bSA/0.jpg)](https://youtu.be/Fu22sGp0bSA)|[![CARLA low resolution](http://img.youtube.com/vi/W5jmReUm7Rg/0.jpg)](https://youtu.be/W5jmReUm7Rg)|
| High Resolution (Ground-Truth)                          | [![KITTI high resolution](http://img.youtube.com/vi/xibLhAfs6nA/0.jpg)](https://youtu.be/xibLhAfs6nA)| [![DurLAR high resolution](http://img.youtube.com/vi/YS4TR6_Kcks/0.jpg)](https://youtu.be/YS4TR6_Kcks)|[![CARLA high resolution](http://img.youtube.com/vi/7GIyPLIMHdc/0.jpg)](https://youtu.be/7GIyPLIMHdc)|
| Implicit LiDAR Network (ILN)                            | [![KITTI iln](http://img.youtube.com/vi/4f8JybY0pag/0.jpg)](https://youtu.be/4f8JybY0pag) | [![DurLAR iln](http://img.youtube.com/vi/79I2n3ALg80/0.jpg)](https://youtu.be/79I2n3ALg80)|[![CARLA iln](http://img.youtube.com/vi/-szKaNGgUsk/0.jpg)](https://youtu.be/-szKaNGgUsk)|
| LiDAR Super-Resolution Network (LiDAR-SR)               | [![KITTI lidar-sr](http://img.youtube.com/vi/u0WA5pUnM6k/0.jpg)](https://youtu.be/u0WA5pUnM6k) | [![DurLAR lidar-sr](http://img.youtube.com/vi/SITqHSFM8f4/0.jpg)](https://youtu.be/SITqHSFM8f4)|[![CARLA lidar-sr](http://img.youtube.com/vi/jX-peCiEv6o/0.jpg)](https://youtu.be/jX-peCiEv6o)|
| ***TULIP***                                             | [![KITTI tulip](http://img.youtube.com/vi/p2-m6vbMH7E/0.jpg)](https://youtu.be/p2-m6vbMH7E) | [![DurLAR tulip](http://img.youtube.com/vi/BZnNUisT70c/0.jpg)](https://youtu.be/BZnNUisT70c)|[![CARLA tulip](http://img.youtube.com/vi/eJ34lr6ZWrc/0.jpg)](https://youtu.be/eJ34lr6ZWrc)|
| ***TULIP-L***                                           | [![KITTI tulip-l](http://img.youtube.com/vi/RvvXSdROFAo/0.jpg)](https://youtu.be/RvvXSdROFAo) | [![DurLAR tulip-l](http://img.youtube.com/vi/RGUL5Pxpz3k/0.jpg)](https://youtu.be/RGUL5Pxpz3k)|[![CARLA tulip-l](http://img.youtube.com/vi/_Qc90E9-gLU/0.jpg)](https://youtu.be/_Qc90E9-gLU)|

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
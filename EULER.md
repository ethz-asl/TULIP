# Guide to setup conda environment on euler cluster

## 1. Download the miniconda package online to the home folder and follow the instructions for installation
https://docs.anaconda.com/free/miniconda/index.html


## 1. Set up python environment in cluster

Python version 3.8 or 3.9 is recommended
```bash
conda create -n myenv python=3.8
```

Open the bashrc file (~/.bashrc) and type the following command

```bash
module load eth_proxy cuda/11.3.1 gcc/8.2.0 ninja
export CUDA_HOME='/cluster/apps/gcc-8.2.0/cuda-11.3.1-o54iuxgz6jm4csvkstuj5hjg4tvd44h3'
```
Note:

* The path to cuda can be different or changed due to the system update, in this case please check it in the euler 

* Remember detach and restart the euler every time after adding some change in .bashrc file 

* You can check the version of those modules by typing

```bash
module list  # all loaded modules
nvcc --version # whether cuda is nvcc-compiled
```

## 2. Install Pytorch and cudatoolkit

Run the following commands to install the packages in your environment

```
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
```

Run `conda list` to check whether pytorch of cuda-version is correctly installed

## 3. Install python packages


```
pip install -r requirements.txt
pip install git+'https://github.com/otaheri/chamfer_distance'  # fast computation for chamfer distance
```

## 4. Start your experiment
```
# interactive job
srun -A ls_riner --gpus=rtx_2080_ti:1 --mem-per-cpu=20G --ntasks=16  --pty bash 
# Submit job to remote server
sbatch --gpus=rtx_3090:1 --mem-per-cpu=20G --ntasks=16 --time=24:00:00 your_task
```
About arguments and commands: https://scicomp.ethz.ch/wiki/Using_the_batch_system \
About gpus in euler cluster: https://scicomp.ethz.ch/wiki/Getting_started_with_GPUs\
Modules in euler cluster can be loaded by calling 'module load your_module'. All available applications: https://scicomp.ethz.ch/wiki/Euler_applications_and_libraries

## Some other notes

You can also use `sbatch` to submit the jobs to remote cluster instead of starting a interactive job. If you want to submit a job with bash script, you have to structure MyJob.sh as following.

```
#!/bin/sh
---- here is your work for submission----
```




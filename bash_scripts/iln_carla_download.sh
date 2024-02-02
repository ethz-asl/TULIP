#!/bin/bash

source $HOME/miniconda3/bin/activate
conda activate plr
module load eth_proxy cuda/11.3.1 gcc/8.2.0 ninja

# wget https://sgvr.kaist.ac.kr/~yskwon/papers/icra22-iln/Carla.zip

unzip Carla.zip
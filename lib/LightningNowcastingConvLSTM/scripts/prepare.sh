#! /bin/bash

mkdir -p data/exp_raw
mkdir -p data/exp_pro

wget --no-show-progress 192.168.1.243:8043/datasets/nowcasting.h5 -P data/exp_raw


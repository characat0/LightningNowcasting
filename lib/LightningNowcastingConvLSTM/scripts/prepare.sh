#! /bin/bash

mkdir -p data/exp_raw
mkdir -p data/exp_pro

BASE_URL=clive.unraid.local:8043

wget --no-show-progress ${BASE_URL}/datasets/nowcasting.h5 -P data/exp_raw
wget --no-show-progress ${BASE_URL}/datasets/train.jld2 -P data/exp_pro
wget --no-show-progress ${BASE_URL}/datasets/val.jld2 -P data/exp_pro

sha256sum -c data/SHA256SUMS

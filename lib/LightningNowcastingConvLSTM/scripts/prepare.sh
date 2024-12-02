#! /bin/bash

mkdir -p data/exp_raw
mkdir -p data/exp_pro

BASE_URL=static.unraid.local:8043

curl -o data/exp_raw/nowcasting.h5 ${BASE_URL}/datasets/nowcasting.h5
curl -o data/exp_pro/train.jld2 ${BASE_URL}/datasets/train.jld2
curl -o data/exp_pro/val.jld2 ${BASE_URL}/datasets/val.jld2

sha256sum -c data/SHA256SUMS

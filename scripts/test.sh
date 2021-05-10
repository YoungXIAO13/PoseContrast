#!/usr/bin/env bash

gpu=0
exp_dir="exps/PoseContrast_Pascal3D_MOCOv2"

python src/test.py --gpu $gpu --dataset Pascal3D --ckpt ${exp_dir}/ckpt.pth
python src/test.py --gpu $gpu --dataset Pix3D --ckpt ${exp_dir}/ckpt.pth


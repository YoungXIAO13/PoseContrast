#!/usr/bin/env bash

gpu=0

exp_dir="exps/PoseContrast_ObjectNet3D_ZeroShot"
python src/test.py --gpu $gpu --dataset ObjectNet3D --ckpt ${exp_dir}/ckpt.pth

exp_dir="exps/PoseContrast_ObjectNet3D_FewShot"
python src/test.py --gpu $gpu --dataset ObjectNet3D --ckpt ${exp_dir}/ckpt.pth

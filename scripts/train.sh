#!/usr/bin/env bash

gpu=0
pretrain="pretrain_models/res50_moco_v2_800ep_pretrain.pth"
exp_dir="exps/PoseContrast_Pascal3D_MOCOv2"

python src/train.py --gpu $gpu --dataset Pascal3D --out ${exp_dir} --pretrain ${pretrain} \
--bs 32 --epochs 15 --lr_step 12 --weighting linear  --poseNCE 1 --tau 0.5



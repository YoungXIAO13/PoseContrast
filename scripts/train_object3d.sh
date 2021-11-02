#!/usr/bin/env bash

gpu=0


# Zero-Shot (first-stage training on ObjectNet3D)
pretrain="pretrain_models/res50_moco_v2_800ep_pretrain.pth"
exp_dir="exps/PoseContrast_ObjectNet3D_ZeroShot"

python src/train.py --gpu $gpu --dataset ObjectNet3D --out ${exp_dir} --pretrain ${pretrain} \
--bs 32 --epochs 10 --lr_step 8 --weighting linear  --poseNCE 1 --tau 0.5


# Few-Shot (second-stage training on ObjectNet3D)
ckpt="exps/PoseContrast_ObjectNet3D_ZeroShot/ckpt.pth"
exp_dir="exps/PoseContrast_ObjectNet3D_FewShot"

python src/train.py --gpu $gpu --dataset ObjectNet3D --out ${exp_dir} --ckpt ${ckpt} \
--bs 32 --epochs 10 --lr_feat 1e-5 --lr_vp 1e-5 --shot 10 \
--weighting linear  --poseNCE 1 --tau 0.5

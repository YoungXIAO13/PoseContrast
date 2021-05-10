# download the res-50 models pre-trained in MOCOv2
wget https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_800ep/moco_v2_800ep_pretrain.pth.tar

# modify the module names
python convert_pretrain.py moco_v2_800ep_pretrain.pth.tar res50_moco_v2_800ep_pretrain.pth

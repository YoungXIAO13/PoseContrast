import argparse
import os, sys
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms

from model.resnet import resnet50
from model.vp_estimator import BaselineEstimator
from dataset.vp_data import normalize
from dataset.data_utils import resize_pad, rotation_err


# =================PARAMETERS=============================== #
parser = argparse.ArgumentParser()

# basic settings
parser.add_argument('--img_path', type=str, default=None)
parser.add_argument('--ckpt', type=str, default=None, help='resume pth')
parser.add_argument('--gpu', type=str, default="0", help='gpu index')

# model hyper-parameters
parser.add_argument('--img_size', type=int, default=224, help='input image size')
parser.add_argument('--img_feature_dim', type=int, default=2048, help='feature dimension for images')
parser.add_argument('--bin_size', type=int, default=15, help='bin size for the euler angle classification')

args = parser.parse_args()
# ========================================================== #


# ================CREATE NETWORK============================ #
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

net_feat = resnet50(num_classes=128)
net_vp = BaselineEstimator(img_feature_dim=args.img_feature_dim)
net_feat.cuda()
net_vp.cuda()

# resume training procedure
if args.ckpt is not None:
    state = torch.load(args.ckpt)
    net_feat.load_state_dict(state['net_feat'])
    net_vp.load_state_dict(state['net_vp'])
else:
    sys.exit(0)

net_feat.eval()
net_vp.eval()
# ========================================================== #

print('Input image: {} \n ----------------------'.format(args.img_path))

im_transform = transforms.Compose([transforms.ToTensor(), normalize])
im = Image.open(args.img_path).convert('RGB')
im_copy = im.copy()
im = resize_pad(im, args.img_size)
im = im_transform(im)
im = im.unsqueeze(0)
im = im.cuda()

with torch.no_grad():
    feat, _ = net_feat(im)
    out = net_vp(feat)
    vp_pred = net_vp.compute_vp_pred(out)

# predictions for original and flipped images
vp_pred = vp_pred.cpu().numpy().squeeze()
vp_pred[1] -= 90
vp_pred[2] -= 180
print('viewpoint prediction:\n Azimuth={}\n Elevation={}\n Inplane Rotation={}'.format(
    vp_pred[0], vp_pred[1], vp_pred[2]))

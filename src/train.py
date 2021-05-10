import argparse
import os, sys
import time
import pickle
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.resnet import resnet50
from model.vp_estimator import BaselineEstimator, Estimator
from dataset.vp_data import Pascal3D
from model.model_utils import save_checkpoint, load_checkpoint, adjust_learning_rate, poseNCE, freeze_model
from dataset.data_utils import AverageValueMeter, rotation_acc, rotation_err
from utils.logger import get_logger
from validation import val


# =================PARAMETERS=============================== #
parser = argparse.ArgumentParser()

# basic settings
parser.add_argument('--flip', default=True, type=bool)
parser.add_argument('--rot', type=int, default=15)
parser.add_argument('--poseNCE', type=float, default=0)
parser.add_argument('--weighting', type=str, default='linear', choices=['linear', 'sqrt', 'square', 'sin', 'sinsin'])
parser.add_argument('--tau', type=float, default=0.5)

parser.add_argument('--dataset', type=str, default='Pascal3D', choices=['ObjectNet3D', 'Pascal3D'])
parser.add_argument('--shot', type=int, default=None, help='K shot number')
parser.add_argument('--pretrain', type=str, default=None, help='pretrain model path')
parser.add_argument('--log', type=str, default="trainer", help='logger name')
parser.add_argument('--out', type=str, default="training", help='output dir')
parser.add_argument('--ckpt', type=str, default=None, help='resume pth')
parser.add_argument('--gpu', type=str, default="0", help='gpu index')

# network training procedure settings
parser.add_argument('--resume', action='store_true')
parser.add_argument('--bs', type=int, default=32, help='batch size')
parser.add_argument('--epochs', type=int, default=15, help='training epochs')
parser.add_argument('--lr_vp', type=float, default=1e-4, help='learning rate of optimizer for viewpoint estimator')
parser.add_argument('--lr_feat', type=float, default=1e-4, help='learning rate of optimizer for backbone')
parser.add_argument('--lr_step', type=int, default=12, help='epoch to decrease')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--print_freq', type=int, default=50, help='frequence of output print')

# model hyper-parameters
parser.add_argument('--img_feature_dim', type=int, default=2048, help='feature dimension for images')
parser.add_argument('--bin_size', type=int, default=15, help='bin size for the euler angle classification')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
# ========================================================== #


# =================CREATE DATASET=========================== #
root_dir = os.path.join('data', args.dataset)
annotation_file = '{}.txt'.format(args.dataset)

if args.dataset == 'Pascal3D':
    dataset_train = Pascal3D(train=True, root_dir=root_dir, annotation_file=annotation_file, rot=args.rot, bs=args.bs)
    dataset_val = Pascal3D(train=False, root_dir=root_dir, annotation_file=annotation_file)

elif args.dataset == 'ObjectNet3D':
    test_classes = ['bed', 'bookshelf', 'calculator', 'cellphone', 'computer', 'door', 'filing_cabinet', 'guitar', 'iron',
                    'knife', 'microwave', 'pen', 'pot', 'rifle', 'shoe', 'slipper', 'stove', 'toilet', 'tub', 'wheelchair']

    exclude_classes = test_classes if args.shot is None else None
    dataset_train = Pascal3D(train=True, root_dir=root_dir, annotation_file=annotation_file,
                             cls_choice=exclude_classes, shot=args.shot, rot=args.rot)
    dataset_val = Pascal3D(train=False, root_dir=root_dir, annotation_file=annotation_file,
                           cls_choice=test_classes)

else:
    raise ValueError

train_loader = DataLoader(dataset_train, batch_size=args.bs, shuffle=True, num_workers=args.workers, drop_last=True)
val_loader = DataLoader(dataset_val, batch_size=args.bs, shuffle=False, num_workers=args.workers)
# ========================================================== #


# ================CREATE NETWORK============================ #
# create feature backbone
net_feat = resnet50(num_classes=128)
if args.pretrain is not None:
    load_checkpoint(net_feat, args.pretrain)

# create viewpoint estimator
net_vp = BaselineEstimator(img_feature_dim=args.img_feature_dim)

net_feat.cuda()
net_vp.cuda()

start_epoch = 0

# resume training procedure
if args.ckpt is not None:
    state = torch.load(args.ckpt)
    net_feat.load_state_dict(state['net_feat'])
    net_vp.load_state_dict(state['net_vp'])
    print('resuming model weights finished!')
    if args.resume:
        start_epoch = state['epoch']
        best_acc = state['best_acc']

# freeze feature backbone if learning rate is set to be zero
if args.lr_feat == 0:
    freeze_model(net_feat)

# create optimizer
optimizer_feat = torch.optim.Adam(net_feat.parameters(), lr=args.lr_feat, weight_decay=0.0005)
optimizer_vp = torch.optim.Adam(net_vp.parameters(), lr=args.lr_vp, weight_decay=0.0005)
# ========================================================== #


# =============DEFINE stuff for logs ======================= #
os.makedirs(args.out, exist_ok=True)
logger = get_logger(args.out, name=args.log)

logger.info(args)
logger.info('\ntraining set: {}'.format(len(dataset_train)))
logger.info('validation set: {}\n'.format(len(dataset_val)))
# ========================================================== #


# =================== DEFINE TRAIN ========================= #
def train(logger, data_loader, net_feat, net_vp, optimizer_feat, optimizer_vp):

    train_loss = AverageValueMeter()
    train_acc_rot = AverageValueMeter()

    # freeze feature backbone if required
    if args.lr_feat != 0:
        net_feat.train()
    else:
        net_feat.eval()

    net_vp.train()

    for i, data in enumerate(data_loader):
        # load data and label
        cls_index, im, label, im_flip, label_flip, im_rot, label_rot, im_pos = data
        im, label = im.cuda(), label.cuda()
        b = im.shape[0]

        # concatenate flipped images
        if args.flip:
            im_flip, label_flip = im_flip.cuda(), label_flip.cuda()
            im = torch.cat((im, im_flip), 0)
            label = torch.cat((label, label_flip), 0)

        # concatenate rotated images
        if args.rot != 0:
            im_rot, label_rot = im_rot.cuda(), label_rot.cuda()
            im = torch.cat((im, im_rot), 0)
            label = torch.cat((label, label_rot), 0)

        # forward pass
        feat, _ = net_feat(im)
        out = net_vp(feat)

        # compute rotation matrix accuracy
        vp_pred = net_vp.compute_vp_pred(out)
        acc_rot = rotation_acc(vp_pred, label.float())
        train_acc_rot.update(acc_rot.item(), im.size(0))

        # viewpoint loss
        loss = net_vp.compute_vp_loss(out, label)

        # contrastive loss
        if args.poseNCE != 0:
            im_pos = im_pos.cuda()
            feat_pos, _ = net_feat(im_pos)
            loss_poseNCE = poseNCE(feat[:b, :], feat_pos, label[:b, :], args.tau, args.weighting)
            loss = loss + args.poseNCE * loss_poseNCE

            if (i + 1) % args.print_freq == 0:
                text = 'Pose NCE loss = {:.2f}'.format(loss_poseNCE.item())
                logger.info(text)
                print(text)


        train_loss.update(loss.item(), im.size(0))

        # compute gradient
        if args.lr_feat != 0:
            optimizer_feat.zero_grad()
            optimizer_vp.zero_grad()
            loss.backward()
            optimizer_vp.step()
            optimizer_feat.step()
        else:
            optimizer_vp.zero_grad()
            loss.backward()
            optimizer_vp.step()

        if (i + 1) % args.print_freq == 0:
            text = 'Epoch-{} -- Iter [{}/{}] loss: {:.2f} || accuracy: {:.2f}'.format(
                epoch, i + 1, len(data_loader), train_loss.avg, train_acc_rot.avg)
            logger.info(text)
            print(text)

    return [train_loss.avg, train_acc_rot.avg]
# ========================================================== #


for epoch in range(start_epoch, args.epochs):
    # update learning rate
    if (epoch + 1) % args.lr_step == 0:
        adjust_learning_rate(optimizer_feat, args.lr_feat)
        adjust_learning_rate(optimizer_vp, args.lr_vp)

    # train
    train_loss, train_acc = train(logger, train_loader, net_feat, net_vp, optimizer_feat, optimizer_vp)

    # evaluate
    eval_acc = val(val_loader, net_feat, net_vp)[0]

    save_checkpoint({
        'epoch': epoch,
        'net_feat': net_feat.state_dict(),
        'net_vp': net_vp.state_dict()
    }, os.path.join(args.out, 'ckpt.pth'))

    text = '\nEpoch-{}: train_loss={:.2f}  ||  train_acc={:.2f} val_acc={:.2f} \n\n'.format(
        epoch, train_loss, train_acc, eval_acc)
    logger.info(text)
    print(text)

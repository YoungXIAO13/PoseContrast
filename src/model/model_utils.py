import os
from collections import OrderedDict as odict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from dataset.data_utils import rotation_err



def adjust_learning_rate(optimizer, base_lr):
    """Sets the learning rate to the initial LR decayed by 10"""
    lr = base_lr * 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def freeze_model(model):
    for m in model.modules():
        for p in m.parameters(): p.requires_grad = False


def freeze_bn(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            for p in m.parameters(): p.requires_grad = False


def save_checkpoint(state, filename):
    """save checkpoint"""
    torch.save(state, filename)


def load_checkpoint(model, pth_file, check=False):
    """load state and network weights"""
    checkpoint = torch.load(pth_file, map_location=lambda storage, loc: storage.cuda())
    if 'model' in checkpoint.keys():
        pretrained_dict = checkpoint['model']
    else:
        pretrained_dict = checkpoint['state_dict']
    model_dict = model.state_dict()

    if check:
        print(len(model_dict.keys()), len(pretrained_dict.keys()))
        missed = [name for name in model_dict.keys() if name not in pretrained_dict.keys()]
        print(missed)

    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'fc' not in k}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print('Pre-trained model weight loaded')


def get_pred_from_cls_output(outputs):
    preds = []
    for n in range(0, len(outputs)):
        output = outputs[n]
        _, pred = output.topk(1, 1, True, True)
        preds.append(pred.view(-1))
    return preds


# classification loss
CE = nn.CrossEntropyLoss().cuda()


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


class SmoothCELoss(nn.Module):
    def __init__(self, range, classes, smooth=0.):
        super(SmoothCELoss, self).__init__()
        self.__range__ = range
        self.__smooth__ = smooth
        self.__SmoothLoss__ = LabelSmoothingLoss(classes, smoothing=smooth, dim=-1)

    def forward(self, pred, target):
        binSize = self.__range__ // pred.size(1)
        trueLabel = target // binSize
        return self.__SmoothLoss__(pred, trueLabel)


def cross_entropy_loss(pred, target, range):
    binSize = range // pred.size(1)
    trueLabel = target // binSize
    return CE(pred, trueLabel)


class ClsLoss(nn.Module):
    def __init__(self, range):
        super(ClsLoss, self).__init__()
        self.__range__ = range
        return

    def forward(self, pred, target):
        return cross_entropy_loss(pred, target, self.__range__)


# regression loss
Huber = nn.SmoothL1Loss().cuda()


def delta_loss(pred_azi, pred_ele, pred_rol, target, bin):
    # compute the ground truth delta value according to angle value and bin size
    target_delta = (target % bin) / bin

    # compute the delta prediction in the ground truth bin
    target_label = (target // bin).long()
    delta_azi = pred_azi[torch.arange(pred_azi.size(0)), target_label[:, 0]]
    delta_ele = pred_ele[torch.arange(pred_ele.size(0)), target_label[:, 1]]
    delta_rol = pred_rol[torch.arange(pred_rol.size(0)), target_label[:, 2]]
    pred_delta = torch.cat((delta_azi.unsqueeze(1), delta_ele.unsqueeze(1), delta_rol.unsqueeze(1)), 1)

    return Huber(5. * pred_delta, 5. * target_delta)


class DeltaLoss(nn.Module):
    def __init__(self, bin):
        super(DeltaLoss, self).__init__()
        self.__bin__ = bin
        return

    def forward(self, pred_azi, pred_ele, pred_rol, target):
        return delta_loss(pred_azi, pred_ele, pred_rol, target, self.__bin__)


class negDotLoss:
    def __init_(self):
        pass

    def compute_loss(self, tgts, Pred, GT):
        Loss = odict()
        for tgt in tgts:
            Loss[tgt] = torch.mean(-torch.bmm(
                GT[tgt].view(GT[tgt].shape[0], 1, 2).float(),
                Pred[tgt].view(Pred[tgt].shape[0], 2, 1).float()))
        return Loss


class CELoss:
    def __init__(self):
        self.CELoss = nn.CrossEntropyLoss().cuda()

    def compute_loss(self, tgts, Pred, GT):
        Loss = odict()
        for tgt in tgts:
            Loss[tgt] = self.CELoss(Pred[tgt].view(Pred[tgt].size()[0], 4), GT[tgt].view(Pred[tgt].size()[0],))
        return Loss


def infoNCE(feat_ori, feat_pos, tau=0.1):
    # L-2 normalization
    feat_ori = F.normalize(feat_ori, dim=-1)
    feat_pos = F.normalize(feat_pos, dim=-1)
    b = feat_pos.shape[0]

    # logits
    l_pos = torch.einsum('nc,nc->n', [feat_ori, feat_pos]).unsqueeze(-1)  # (N,1)
    l_neg = torch.einsum('nc,ck->nk', [feat_ori, feat_ori.transpose(0, 1)])  # (N, N)
    logits = (1 - torch.eye(b)).type_as(l_neg) * l_neg + torch.eye(b).type_as(l_pos) * l_pos

    # loss
    logits = logits / tau
    labels = torch.arange(b, dtype=torch.long).cuda()
    loss = F.cross_entropy(logits, labels)
    return loss


def poseNCE(feat_ori, feat_pos, label, tau=0.1, weighting="linear"):
    # L-2 normalization
    feat_ori = F.normalize(feat_ori, dim=-1)
    feat_pos = F.normalize(feat_pos, dim=-1)
    feat_all = feat_ori.clone()
    label_all = label.clone()
    b = feat_ori.shape[0]
    
    # compute pose-distance in adjacency (N, N)
    label_ori_rep = label.reshape(-1, 1, 3).repeat(1, b, 1)
    label_all_rep = label_all.reshape(1, -1, 3).repeat(b, 1, 1)
    dist = rotation_err(label_ori_rep.reshape(-1, 3), label_all_rep.reshape(-1, 3))
    dist = dist.reshape(b, b)

    # rescale the dist from [0, 180] degrees to [0, 1] 
    if weighting == 'linear':
        dist = dist / 180
    elif weighting == 'square':
        dist = (dist / 180) ** 2
    elif weighting == 'sqrt':
        dist = torch.sqrt(dist / 180)
    elif weighting == 'sin':
        dist = torch.abs(torch.sin(dist / 180 * np.pi))
    elif weighting == 'sinsin':
        dist = torch.sin(dist / 180 * np.pi) ** 2

    # logites
    l_pos = torch.exp(torch.einsum('nc,nc->n', [feat_ori, feat_pos]).unsqueeze(-1) / tau)  # (N,1)
    l_neg = torch.exp(torch.einsum('nc,ck->nk', [feat_ori, feat_all.transpose(0, 1)]) / tau) * dist  # (N, N)
    
    logits = torch.cat([l_pos, l_neg], dim=1)

    # loss
    loss = - torch.log(logits[:, 0] / torch.sum(logits, -1))
    return loss.mean()



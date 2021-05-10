from collections import OrderedDict as odict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .model_utils import ClsLoss, DeltaLoss, negDotLoss, CELoss, SmoothCELoss


class BaselineEstimator(nn.Module):
    def __init__(self, img_feature_dim=1024, azi_classes=24, ele_classes=12, inp_classes=24, bin_size=15):
        super(BaselineEstimator, self).__init__()
        self.bin_size = bin_size

        # Pose estimator
        feat_dim = 200
        self.compress = nn.Sequential(
            nn.Linear(img_feature_dim, feat_dim*4), nn.BatchNorm1d(feat_dim*4), nn.ReLU(inplace=True),
            nn.Linear(feat_dim*4, feat_dim*2), nn.BatchNorm1d(feat_dim*2), nn.ReLU(inplace=True),
            nn.Linear(feat_dim*2, feat_dim), nn.BatchNorm1d(feat_dim), nn.ReLU(inplace=True)
        )
        self.fc_cls_azi = nn.Linear(feat_dim, azi_classes)
        self.fc_cls_ele = nn.Linear(feat_dim, ele_classes)
        self.fc_cls_inp = nn.Linear(feat_dim, inp_classes)
        self.fc_reg_azi = nn.Linear(feat_dim, azi_classes)
        self.fc_reg_ele = nn.Linear(feat_dim, ele_classes)
        self.fc_reg_inp = nn.Linear(feat_dim, inp_classes)

        # Setup the loss here.
        self.loss_reg = DeltaLoss(bin_size)
        self.loss_cls_azi = ClsLoss(360)
        self.loss_cls_ele = ClsLoss(180)
        self.loss_cls_inp = ClsLoss(360)


    def forward(self, x):
        x = self.compress(x)

        cls_azi = self.fc_cls_azi(x)
        cls_ele = self.fc_cls_ele(x)
        cls_inp = self.fc_cls_inp(x)

        reg_azi = self.fc_reg_azi(x)
        reg_ele = self.fc_reg_ele(x)
        reg_inp = self.fc_reg_inp(x)
        return [cls_azi, cls_ele, cls_inp, reg_azi, reg_ele, reg_inp]

    def compute_vp_loss(self, out, label):
        loss_azi = self.loss_cls_azi(out[0], label[:, 0])
        loss_ele = self.loss_cls_ele(out[1], label[:, 1])
        loss_inp = self.loss_cls_inp(out[2], label[:, 2])
        loss_reg = self.loss_reg(out[3], out[4], out[5], label.float())
        loss = loss_azi + loss_ele + loss_inp + loss_reg
        return loss

    def compute_vp_pred(self, out, return_scores=False):
        # get predictions for the three Euler angles
        preds = []
        scores = []
        for n in range(3):
            out_cls = out[n]
            _, pred_cls = out_cls.topk(1, 1, True, True)
            pred_cls = pred_cls.view(-1)

            out_reg = out[n + 3]
            pred_reg = out_reg[torch.arange(out_reg.size(0)), pred_cls.long()]

            preds.append((pred_cls.float() + pred_reg) * self.bin_size)

            if return_scores:
                scores.append(out_cls.softmax(-1).max(-1)[0].reshape(-1, 1))

        vp_pred = torch.cat((preds[0].unsqueeze(1), preds[1].unsqueeze(1), preds[2].unsqueeze(1)), 1)
        vp_pred = torch.clamp(vp_pred, min=0, max=360)

        if return_scores:
            scores = torch.cat(scores, 1)
            return vp_pred, scores
        else:
            return vp_pred


class Estimator(nn.Module):
    def __init__(self, img_feature_dim=1024, azi_classes=24, ele_classes=12, inp_classes=24, bin_size=15):
        super(Estimator, self).__init__()
        self.bin_size = bin_size

        # Pose estimator
        feat_dim = 512
        self.project_azi = nn.Sequential(
            nn.Linear(img_feature_dim, feat_dim), nn.BatchNorm1d(feat_dim), nn.ReLU(inplace=True),
        )
        self.project_ele = nn.Sequential(
            nn.Linear(img_feature_dim, feat_dim), nn.BatchNorm1d(feat_dim), nn.ReLU(inplace=True),
        )
        self.project_inp = nn.Sequential(
            nn.Linear(img_feature_dim, feat_dim), nn.BatchNorm1d(feat_dim), nn.ReLU(inplace=True),
        )
        self.fc_cls_azi = nn.Linear(feat_dim, azi_classes)
        self.fc_cls_ele = nn.Linear(feat_dim, ele_classes)
        self.fc_cls_inp = nn.Linear(feat_dim, inp_classes)
        self.fc_reg_azi = nn.Linear(feat_dim, azi_classes)
        self.fc_reg_ele = nn.Linear(feat_dim, ele_classes)
        self.fc_reg_inp = nn.Linear(feat_dim, inp_classes)

        # Setup the loss here.
        self.loss_reg = DeltaLoss(bin_size)
        self.loss_cls_azi = ClsLoss(360)
        self.loss_cls_ele = ClsLoss(180)
        self.loss_cls_inp = ClsLoss(360)

    def forward(self, x):
        x_azi = self.project_azi(x)
        x_ele = self.project_azi(x)
        x_inp = self.project_azi(x)

        cls_azi = self.fc_cls_azi(x_azi)
        cls_ele = self.fc_cls_ele(x_ele)
        cls_inp = self.fc_cls_inp(x_inp)

        reg_azi = self.fc_reg_azi(x_azi)
        reg_ele = self.fc_reg_ele(x_ele)
        reg_inp = self.fc_reg_inp(x_inp)
        return [cls_azi, cls_ele, cls_inp, reg_azi, reg_ele, reg_inp]

    def compute_vp_loss(self, out, label):
        loss_azi = self.loss_cls_azi(out[0], label[:, 0])
        loss_ele = self.loss_cls_ele(out[1], label[:, 1])
        loss_inp = self.loss_cls_inp(out[2], label[:, 2])
        loss_reg = self.loss_reg(out[3], out[4], out[5], label.float())
        loss = loss_azi + loss_ele + loss_inp + loss_reg
        return loss

    def compute_vp_pred(self, out, return_scores=False):
        # get predictions for the three Euler angles
        preds = []
        scores = []
        for n in range(3):
            out_cls = out[n]
            _, pred_cls = out_cls.topk(1, 1, True, True)
            pred_cls = pred_cls.view(-1)

            out_reg = out[n + 3]
            pred_reg = out_reg[torch.arange(out_reg.size(0)), pred_cls.long()]

            preds.append((pred_cls.float() + pred_reg) * self.bin_size)

            if return_scores:
                scores.append(out_cls.softmax(-1).max(-1)[0].reshape(-1, 1))

        vp_pred = torch.cat((preds[0].unsqueeze(1), preds[1].unsqueeze(1), preds[2].unsqueeze(1)), 1)
        vp_pred = torch.clamp(vp_pred, min=0, max=360)

        if return_scores:
            scores = torch.cat(scores, 1)
            return vp_pred, scores
        else:
            return vp_pred



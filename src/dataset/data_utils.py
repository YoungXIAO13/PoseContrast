import os
import numpy as np
import shutil
from PIL import Image
import torch
import torchvision.transforms as transforms


def process_viewpoint_label(label, offset=0):
    label[0] = (360. - label[0] + offset) % 360.
    label[1] = label[1] + 90.
    label[2] = (label[2] + 180.) % 360.
    label = label.astype('int')
    label = torch.from_numpy(label).long()
    return label


# Lighting noise transform
class TransLightning(object):
    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone()\
            .mul(alpha.view(1, 3).expand(3, 3))\
            .mul(self.eigval.view(1, 3).expand(3, 3))\
            .sum(1).squeeze()
        return img.add(rgb.view(3, 1, 1).expand_as(img))


class AverageValueMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.avg = self.avg * (self.count / (self.count + n)) + val * (n / (self.count + n))
        self.count += n


# Crop the image using random bounding box with IoU >= 0.7 compared with the ground truth
def random_crop(im, x, y, w, h):
    left = max(0, x + int(np.random.uniform(-0.1, 0.1) * w))
    upper = max(0, y + int(np.random.uniform(-0.1, 0.1) * h))
    right = min(im.size[0], x + int(np.random.uniform(0.9, 1.1) * w))
    lower = min(im.size[1], y + int(np.random.uniform(0.9, 1.1) * h))
    im_crop = im.crop((left, upper, right, lower))
    return im_crop


def resize_pad(im, dim):
    w, h = im.size
    im = transforms.functional.resize(im, int(dim * min(w, h) / max(w, h)))
    left = int(np.ceil((dim - im.size[0]) / 2))
    right = int(np.floor((dim - im.size[0]) / 2))
    top = int(np.ceil((dim - im.size[1]) / 2))
    bottom = int(np.floor((dim - im.size[1]) / 2))
    im = transforms.functional.pad(im, (left, top, right, bottom))
    return im


def resize_pad_PIL(im, desired_size, mode="RGB"):
    # compute the new size
    old_size = im.size
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    im = im.resize(new_size, Image.ANTIALIAS)

    # create a new image and paste the resized on it
    new_im = Image.new(mode, (desired_size, desired_size))
    new_im.paste(im, ((desired_size - new_size[0]) // 2, (desired_size - new_size[1]) // 2))
    return new_im


def angles_to_matrix(angles):
    """Compute the rotation matrix from euler angles for a mini-batch.
    This is a PyTorch implementation computed by myself for calculating
    R = Rz(inp) Rx(ele - pi/2) Rz(-azi)
    
    For the original numpy implementation in StarMap, you can refer to:
    https://github.com/xingyizhou/StarMap/blob/26223a6c766eab3c22cddae87c375150f84f804d/tools/EvalCls.py#L20
    """
    azi = angles[:, 0]
    ele = angles[:, 1]
    rol = angles[:, 2]
    element1 = (torch.cos(rol) * torch.cos(azi) - torch.sin(rol) * torch.cos(ele) * torch.sin(azi)).unsqueeze(1)
    element2 = (torch.sin(rol) * torch.cos(azi) + torch.cos(rol) * torch.cos(ele) * torch.sin(azi)).unsqueeze(1)
    element3 = (torch.sin(ele) * torch.sin(azi)).unsqueeze(1)
    element4 = (-torch.cos(rol) * torch.sin(azi) - torch.sin(rol) * torch.cos(ele) * torch.cos(azi)).unsqueeze(1)
    element5 = (-torch.sin(rol) * torch.sin(azi) + torch.cos(rol) * torch.cos(ele) * torch.cos(azi)).unsqueeze(1)
    element6 = (torch.sin(ele) * torch.cos(azi)).unsqueeze(1)
    element7 = (torch.sin(rol) * torch.sin(ele)).unsqueeze(1)
    element8 = (-torch.cos(rol) * torch.sin(ele)).unsqueeze(1)
    element9 = (torch.cos(ele)).unsqueeze(1)
    return torch.cat((element1, element2, element3, element4, element5, element6, element7, element8, element9), dim=1)


def rotation_err(preds, targets):
    """compute rotation error for viewpoint estimation"""
    preds = preds.float().clone()
    targets = targets.float().clone()
    
    # get elevation and inplane-rotation in the right format
    # R = Rz(inp) Rx(ele - pi/2) Rz(-azi)
    preds[:, 1] = preds[:, 1] - 180.
    preds[:, 2] = preds[:, 2] - 180.
    targets[:, 1] = targets[:, 1] - 180.
    targets[:, 2] = targets[:, 2] - 180.
    
    # change degrees to radians
    preds = preds * np.pi / 180.
    targets = targets * np.pi / 180.
    
    # get rotation matrix from euler angles
    R_pred = angles_to_matrix(preds)
    R_gt = angles_to_matrix(targets)
    
    # compute the angle distance between rotation matrix in degrees
    R_err = torch.acos(((torch.sum(R_pred * R_gt, 1)).clamp(-1., 3.) - 1.) / 2)
    R_err = R_err * 180. / np.pi
    return R_err


def rotation_acc(preds, targets, th=30.):
    R_err = rotation_err(preds, targets)
    return 100. * torch.mean((R_err <= th).float())


def angle_err(preds, targets):
    """compute rotation error for viewpoint estimation"""
    errs = torch.abs(preds - targets)
    errs = torch.min(errs, 360. - errs)
    return errs


if __name__ == '__main__':
    a = torch.rand(4, 3) * 360
    a[:, 1] = a[:, 1] / 2
    a = a.int()
    b = a.clone()
    b[:, 0] = (b[:, 0] + 180) % 360
    err = rotation_err(a, b)
    print(a, '\n', b, '\n', err)

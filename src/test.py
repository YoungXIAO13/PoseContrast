import argparse
import os, sys
import numpy as np
from tqdm import tqdm
import pandas as pd
import pickle

import torch
from torch.utils.data import DataLoader

from model.resnet import resnet50
from model.vp_estimator import BaselineEstimator, Estimator
from dataset.vp_data import Pascal3D
from dataset.data_utils import AverageValueMeter, rotation_acc, rotation_err
from utils.logger import get_logger
from validation import val

# =================PARAMETERS=============================== #
parser = argparse.ArgumentParser()

# basic settings
parser.add_argument('--dataset', type=str, default='Pascal3D')
parser.add_argument('--log', type=str, default="tester", help='logger name')
parser.add_argument('--out', type=str, default=None, help='output dir')
parser.add_argument('--ckpt', type=str, default=None, help='resume pth')
parser.add_argument('--gpu', type=str, default="0", help='gpu index')
parser.add_argument('--AggFlip', action='store_true')

# network training procedure settings
parser.add_argument('--bs', type=int, default=128, help='batch size')
parser.add_argument('--epochs', type=int, default=200, help='training epochs')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)

# model hyper-parameters
parser.add_argument('--img_feature_dim', type=int, default=2048, help='feature dimension for images')
parser.add_argument('--bin_size', type=int, default=15, help='bin size for the euler angle classification')

args = parser.parse_args()

if args.out is None:
    args.out = os.path.dirname(args.ckpt)
# ========================================================== #


# =================== DEFINE TEST ========================== #
def test_category(val_loader, net_feat, net_vp, out, logger, cls, AggFlip=False):
    _, predictions, labels, scores = val(val_loader, net_feat, net_vp, AggFlip)

    # save predictions
    out_file = os.path.join(out, 'prediction', '{}.npy'.format(cls))
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    np.save(out_file, predictions.cpu().numpy())

    # calculate the rotation errors between prediction and ground truth
    test_errs = rotation_err(predictions, labels.float()).cpu().numpy()
    Acc = np.mean(test_errs <= 30)
    Med = np.median(test_errs)

    scores = scores.cpu().numpy()
    correlation = {"err": test_errs, "scores": scores}
    out_file = os.path.join(out, 'correlation', '{}.pkl'.format(cls))
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, 'wb') as f:
        pickle.dump(correlation, f, protocol=4)

    text = 'Acc_pi/6 is {:.2f} and Med_Err is {:.2f} for class {}\n'.format(Acc, Med, cls)
    logger.info(text)
    print(text)

    return Acc, Med, test_errs
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
# ========================================================== #


# =============DEFINE stuff for logs ======================= #
os.makedirs(args.out, exist_ok=True)

pred_dir = os.path.join(args.out, 'Evaluation_{}'.format(args.dataset))
os.makedirs(pred_dir, exist_ok=True)

logger = get_logger(pred_dir, name=args.log)
logger.info(args)
# ========================================================== #


root_dir = os.path.join('data', args.dataset)
annotation_file = '{}.txt'.format(args.dataset)
Accs, Meds = [], []
Errs = []


if args.dataset == 'Pascal3D':
    test_classes = ['aeroplane', 'bicycle', 'boat', 'bottle', 'bus', 'car',
                    'chair', 'diningtable', 'motorbike', 'sofa', 'train', 'tvmonitor']

    for cls in test_classes:
        dataset_val = Pascal3D(train=False, root_dir=root_dir, annotation_file=annotation_file, cls_choice=[cls])
        val_loader = DataLoader(dataset_val, batch_size=args.bs, shuffle=False, num_workers=args.workers)
        print('Tested on %d images of category %s' % (len(dataset_val), cls))

        cls_accs, cls_meds, cls_errs = test_category(val_loader, net_feat, net_vp, pred_dir, logger, cls, args.AggFlip)
        Accs.append(cls_accs)
        Meds.append(cls_meds)
        Errs.extend(cls_errs)

elif args.dataset == 'ObjectNet3D':
    test_classes = ['bed', 'bookshelf', 'calculator', 'cellphone', 'computer', 'door', 'filing_cabinet', 'guitar', 'iron',
                    'knife', 'microwave', 'pen', 'pot', 'rifle', 'shoe', 'slipper', 'stove', 'toilet', 'tub', 'wheelchair']
    
    df = pd.read_csv(os.path.join(root_dir, annotation_file))
    all_classes = np.unique(df.cls_name)
    base_classes = [c for c in all_classes if c not in test_classes]
    all_classes = base_classes + test_classes

    for cls in all_classes:
        dataset_val = Pascal3D(train=False, root_dir=root_dir, annotation_file=annotation_file, cls_choice=[cls])
        val_loader = DataLoader(dataset_val, batch_size=args.bs, shuffle=False, num_workers=args.workers)

        print('Tested on %d images of category %s' % (len(dataset_val), cls))
        cls_accs, cls_meds, cls_errs = test_category(val_loader, net_feat, net_vp, pred_dir, logger, cls, args.AggFlip)
        Accs.append(cls_accs)
        Meds.append(cls_meds)
        Errs.extend(cls_errs)

    text = '\nAverage for Base Class  >>>>  Acc_pi/6 is {:.2f} and Med_Err is {:.2f}'.format(
        np.mean(Accs[:80]), np.mean(Meds[:80])
    )
    logger.info(text)
    print(text)

    text = 'Average for Novel Class  >>>>  Acc_pi/6 is {:.2f} and Med_Err is {:.2f}\n'.format(
        np.mean(Accs[80:]), np.mean(Meds[80:])
    )
    logger.info(text)
    print(text)

elif args.dataset == 'Pix3D':
    test_classes = ['tool', 'misc', 'bookcase', 'wardrobe', 'desk', 'bed', 'table', 'sofa', 'chair']

    for cls in tqdm(test_classes):
        dataset_val = Pascal3D(train=False, root_dir=root_dir, annotation_file=annotation_file, cls_choice=[cls])
        val_loader = DataLoader(dataset_val, batch_size=args.bs, shuffle=False, num_workers=args.workers)

        print('Tested on %d images of category %s' % (len(dataset_val), cls))
        cls_accs, cls_meds, cls_errs = test_category(val_loader, net_feat, net_vp, pred_dir, logger, cls, args.AggFlip)
        Accs.append(cls_accs)
        Meds.append(cls_meds)
        Errs.extend(cls_errs)

else:
    sys.exit(0)

text = '\nAverage for All Classes  >>>>  Acc_pi/6 is {:.2f} and Med_Err is {:.2f}'.format(
    np.mean(Accs), np.mean(Meds)
)
logger.info(text)
print(text)

text = 'Average for All Samples  >>>>  Acc_pi/6 is {:.2f} and Med_Err is {:.2f}'.format(
    np.mean(np.array(Errs) <= 30), np.median(np.array(Errs))
)
logger.info(text)
print(text)



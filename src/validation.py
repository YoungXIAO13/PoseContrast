import torch

from dataset.data_utils import AverageValueMeter, rotation_acc, rotation_err


def val(data_loader, net_feat, net_vp, AggFlip=False):

    val_acc_rot = AverageValueMeter()
    predictions = []
    labels = []
    inconsis = []
    scores = []

    net_feat.eval()
    net_vp.eval()

    with torch.no_grad():
        for i, data in enumerate(data_loader):
            # load data and label
            cls_index, im, label, im_flip = data
            im, label = im.cuda(), label.cuda()
            cls_index = cls_index.cuda()

            # forward pass
            feat, _ = net_feat(im)
            out = net_vp(feat)
            vp_pred, score = net_vp.compute_vp_pred(out, True)
            scores.append(score)

            # test-time aug with flipped image
            if AggFlip:
                im_flip = im_flip.cuda()
                feat_flip, _ = net_feat(im_flip)
                out_flip = net_vp(feat_flip)
                vp_pred_flip, score_flip = net_vp.compute_vp_pred(out_flip, True)
                vp_pred_flip[:, 0] = 360 - vp_pred_flip[:, 0]
                vp_pred_flip[:, 2] = 360 - vp_pred_flip[:, 2]
                
                vp_pred_ori_flip = torch.cat((vp_pred.unsqueeze(-1), vp_pred_flip.unsqueeze(-1)), -1)
                azi_score = torch.cat((score[:, 0].unsqueeze(-1), score_flip[:, 0].unsqueeze(-1)), -1)
                select = azi_score.max(-1)[-1]
                vp_pred = torch.gather(vp_pred_ori_flip, -1, select.view(-1, 1, 1).expand(label.shape[0], 3, 1)).squeeze()
            
            # compute accuracy
            acc_rot = rotation_acc(vp_pred, label.float())
            val_acc_rot.update(acc_rot.item(), im.size(0))

            # append results and labels
            predictions.append(vp_pred)
            labels.append(label)

    predictions = torch.cat(predictions, dim=0)
    labels = torch.cat(labels, dim=0)
    scores = torch.cat(scores, dim=0)

    return val_acc_rot.avg, predictions, labels, scores

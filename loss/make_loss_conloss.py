# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from .triplet_loss import TripletLoss
from .center_loss import CenterLoss
from .multi_modal_margin_loss_new2 import multiModalMarginLossNew as multiModalMarginLossNew2

def make_loss(cfg, num_classes):    # modified by gu
    loss_type = cfg.MODEL.LOSS_TYPE
    sampler = cfg.DATALOADER.SAMPLER
    dataset = cfg.DATASETS.NAMES
    feat_dim = 2048
    criterion_m = multiModalMarginLossNew2(margin=cfg.MODEL.MMM_LOSS_MARGIN)
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss
    if 'triplet' in cfg.MODEL.METRIC_LOSS_TYPE:
        if cfg.MODEL.NO_MARGIN:
            triplet = TripletLoss()
            print("using soft triplet loss for training")
        else:
            triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
            print("using triplet loss with margin:{}".format(cfg.SOLVER.MARGIN))
    else:
        print('expected METRIC_LOSS_TYPE should be triplet'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        print("label smooth on, numclasses:", num_classes)

    if sampler == 'softmax':
        def loss_func(score, feat, target):
            return F.cross_entropy(score, target)

    elif cfg.DATALOADER.SAMPLER == 'softmax_triplet':
        def loss_func(score, feat, target, target_cam):
            if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    if isinstance(score, list):
                        ID_LOSS = [xent(scor, target) for scor in score[1:]]
                        ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                        ID_LOSS = 0.5 * ID_LOSS + 0.5 * xent(score[0], target)
                    else:
                        ID_LOSS = xent(score, target)

                    if isinstance(feat, list):
                        TRI_LOSS = [triplet(feats, target)[0] for feats in feat[1:]]
                        TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
                        TRI_LOSS = 0.5 * TRI_LOSS + 0.5 * triplet(feat[0], target)[0]
                    else:
                        TRI_LOSS = triplet(feat, target)[0]
                            
                    loss_item_list = [ID_LOSS.item(), TRI_LOSS.item()]
                    return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + \
                               cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS, loss_item_list
                else:
                    if isinstance(score, list):
                        ID_LOSS = [F.cross_entropy(scor, target) for scor in score[1:]]
                        ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                        ID_LOSS = 0.5 * ID_LOSS + 0.5 * F.cross_entropy(score[0], target)
                    else:
                        ID_LOSS = F.cross_entropy(score, target)

                    if isinstance(feat, list):
                        if '3M' in loss_type:
                            MMM_LOSS = criterion_m(F.normalize(feat[1], p=2, dim=1), F.normalize(feat[2], p=2, dim=1), F.normalize(feat[3], p=2, dim=1), target)
                            if dataset in ['RGBNT201']:
                                TRI_LOSS = triplet(feat[0], target)[0]
                            else:
                                TRI_LOSS = [triplet(feats, target)[0] for feats in feat]
                                TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
                        elif '3M_F' in loss_type:
                            MMM_LOSS = criterion_m(F.normalize(feat[2], p=2, dim=1), F.normalize(feat[3], p=2, dim=1), F.normalize(feat[4], p=2, dim=1), target)
                            TRI_LOSS = triplet(feat[0], target)[0] + triplet(feat[1], target)[0]
                        else:
                            TRI_LOSS = [triplet(feats, target)[0] for feats in feat]
                            TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
                        # TRI_LOSS = 0.5 * TRI_LOSS + 0.5 * triplet(feat[0], target)[0]
                    else:
                            TRI_LOSS = triplet(feat, target)[0]

                    if '3M' in loss_type: 
                        loss_item_list = [ID_LOSS.item(), TRI_LOSS.item(), MMM_LOSS.item()]
                        # loss_item_list = [ID_LOSS.item(), TRI_LOSS.item()]
                        return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + \
                                cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS + cfg.MODEL.MMM_LOSS_WEIGHT * MMM_LOSS, loss_item_list
                    else:
                        loss_item_list = [ID_LOSS.item(), TRI_LOSS.item()]
                        return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + \
                                cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS, loss_item_list
            else:
                print('expected METRIC_LOSS_TYPE should be triplet'
                      'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    else:
        print('expected sampler should be softmax, triplet, softmax_triplet or softmax_triplet_center'
              'but got {}'.format(cfg.DATALOADER.SAMPLER))
    return loss_func, center_criterion




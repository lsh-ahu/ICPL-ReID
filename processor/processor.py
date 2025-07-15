import logging
import os
import collections
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from torch.cuda import amp
import torch.distributed as dist
from loss.supcontrast import SupConLoss
import numpy
from datetime import timedelta
import numpy as np
from ClusterContrast.cm import ClusterMemory
from ClusterContrast.cm_300 import ClusterMemory_300
import torch.nn.functional as F
from utils.logger import PrettyEvaluation

def do_train(cfg,
             model,
             center_criterion,
             train_loader,
             train_loader_normal,
             val_loader,
             cam_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query, local_rank):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    eval_logger = PrettyEvaluation(cfg, num_query)

    logger = logging.getLogger("ICPL.image_train")
    logger_eval = logging.getLogger("evaluator")
    logger_eval_bn = logging.getLogger("evaluator_bn")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1:
            if cfg.MODEL.DIST_TRAIN:
                print('Using {} GPUs for training'.format(torch.cuda.device_count()))
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
            else:
                print('Using {} GPUs for training'.format(torch.cuda.device_count()))
                model = nn.DataParallel(model)

    loss_meter = AverageMeter()
    loss_meter1 = AverageMeter()
    loss_meter2 = AverageMeter()
    loss_meter3 = AverageMeter()
    loss_meter4 = AverageMeter()
    loss_meter5 = AverageMeter()
    loss_meter6 = AverageMeter()
    loss_meter7 = AverageMeter()
    loss_meter8 = AverageMeter()
    loss_meter9 = AverageMeter()
    loss_meter10 = AverageMeter()
    loss_meter11 = AverageMeter()
    loss_meter12 = AverageMeter()
    acc_meter = AverageMeter()
    acc_meter1 = AverageMeter()
    acc_meter2 = AverageMeter()

    scaler = amp.GradScaler()
    xent = SupConLoss(device)

    all_start_time = time.monotonic()

    for epoch in range(1, epochs + 1):
        model.train()
        logger.info('start extract prototype features.')
        image_features1 = []
        image_features2 = []
        image_features3 = []
        labels = []
        with torch.no_grad():
            for n_iter, (img1, img2, img3, vid, target_cam, scene) in enumerate(train_loader_normal):
                img1 = img1.to(device)
                img2 = img2.to(device)
                if cfg.DATASETS.NAMES != 'RGBN300':
                    img3 = img3.to(device)
                target = vid.to(device)
                with amp.autocast(enabled=True):
                    if epoch <= 3:
                        peft_stage = False
                    else:
                        peft_stage = True
                    if cfg.DATASETS.NAMES != 'RGBN300':
                        image_feature1, image_feature2, image_feature3 = model(img1, img2, img3, label = target, get_image=True, peft_stage=peft_stage)
                        for i, img_feat1, img_feat2, img_feat3 in zip(target, image_feature1, image_feature2, image_feature3):
                            labels.append(i)
                            image_features1.append(img_feat1.cpu())
                            image_features2.append(img_feat2.cpu())
                            image_features3.append(img_feat3.cpu())
                    else:
                        image_feature1, image_feature2 = model(img1, img2, label = target)
                        for i, img_feat1, img_feat2 in zip(target, image_feature1[1][1], image_feature2[1][1]):
                            labels.append(i)
                            image_features1.append(img_feat1.cpu())
                            image_features2.append(img_feat2.cpu())
            labels_list = torch.stack(labels, dim=0).cuda() #N
            image_features_list1 = torch.stack(image_features1, dim=0).cuda()
            image_features_list2 = torch.stack(image_features2, dim=0).cuda()
            if cfg.DATASETS.NAMES != 'RGBN300':
                image_features_list3 = torch.stack(image_features3, dim=0).cuda()

        # generate new dataset and calculate cluster centers
        @torch.no_grad()
        def generate_cluster_features(labels, features):
            centers = collections.defaultdict(list)
            for i, label in enumerate(labels):
                label = label.item()
                if label == -1:
                    continue
                centers[label].append(features[i])

            centers = [
                torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
            ]

            centers = torch.stack(centers, dim=0)
            return centers

        cluster_features_rgb = generate_cluster_features(labels_list, image_features_list1)
        cluster_features_nir = generate_cluster_features(labels_list, image_features_list2)
        if cfg.DATASETS.NAMES != 'RGBN300':
            cluster_features_tir = generate_cluster_features(labels_list, image_features_list3)

        if cfg.DATASETS.NAMES != 'RGBN300':
            del image_features_list1, image_features_list2, image_features_list3
        else:
            del image_features_list1, image_features_list2

        num_cluster = cluster_features_rgb.shape[0]
        if cfg.DATASETS.NAMES != 'RGBN300':
            memory = ClusterMemory(768, num_cluster, temp=cfg.MODEL.CM.TEMP,
                                momentum=cfg.MODEL.CM.MOMENTUM, use_hard=cfg.MODEL.CM.USE_HARD, change_scale=cfg.MODEL.CM.CHANGE_SCALE).cuda()
        else:
            memory = ClusterMemory_300(768, num_cluster, temp=cfg.MODEL.CM.TEMP,
                                momentum=cfg.MODEL.CM.MOMENTUM, use_hard=cfg.MODEL.CM.USE_HARD, change_scale=cfg.MODEL.CM.CHANGE_SCALE).cuda()
            
        memory.features_rgb = F.normalize(cluster_features_rgb, dim=1).cuda()
        memory.features_nir = F.normalize(cluster_features_nir, dim=1).cuda()
        if cfg.DATASETS.NAMES != 'RGBN300':
            memory.features_tir = F.normalize(cluster_features_tir, dim=1).cuda()
        logger.info('end extract prototype features.')

        start_time = time.time()
        loss_meter.reset()
        loss_meter1.reset()
        loss_meter2.reset()
        loss_meter3.reset()
        loss_meter4.reset()
        loss_meter5.reset()
        loss_meter6.reset()
        loss_meter7.reset()
        loss_meter8.reset()
        loss_meter9.reset()
        loss_meter10.reset()
        loss_meter11.reset()
        loss_meter12.reset()
        acc_meter.reset()
        acc_meter1.reset()
        acc_meter2.reset()

        eval_logger.reset()
        scheduler.step(epoch)

        image_features1 = []
        image_features2 = []
        image_features3 = []
        labels = []
        
        for n_iter, (img1, img2, img3, vid, target_cam, scene) in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img1 = img1.to(device)
            img2 = img2.to(device)
            if cfg.DATASETS.NAMES != 'RGBN300':
                img3 = img3.to(device)
            target = vid.to(device)
            if cfg.MODEL.SIE_CAMERA:
                target_cam = target_cam.to(device)
            else: 
                target_cam = None
            if cfg.MODEL.SIE_VIEW:
                target_view = target_view.to(device)
            else: 
                target_view = None
            with amp.autocast(enabled=True):
                if cfg.DATASETS.NAMES != 'RGBN300':
                    mode1, mode2, mode3 = model(img1, img2, img3, label=target, cam_label=target_cam)
                else:
                    mode1, mode2 = model(img1, img2, label=target, cam_label=target_cam)

                # 1 image align to prototype
                if cfg.DATASETS.NAMES != 'RGBN300':
                    loss_i2p_rgb, loss_i2p_nir, loss_i2p_tir = memory(mode1[1][1], mode2[1][1], mode3[1][1], target)
                else:
                    loss_i2p_rgb, loss_i2p_nir = memory(mode1[1][1], mode2[1][1], target)

                # 2 text prompt align to prototype
                target_uni = target.unique()
                if cfg.DATASETS.NAMES != 'RGBN300':
                    text_features1, text_features2, text_features3 = model(label = target_uni, get_text = True)
                else:
                    text_features1, text_features2 = model(label = target_uni, get_text = True)
                
                rgb_proto = memory.features_rgb[target_uni]
                nir_proto = memory.features_nir[target_uni]
                if cfg.DATASETS.NAMES != 'RGBN300':
                    tir_proto = memory.features_tir[target_uni]
                with torch.no_grad():
                    if cfg.DATASETS.NAMES != 'RGBN300':
                        rgb_proto, nir_proto, tir_proto = model(x1_proto=rgb_proto, x2_proto=nir_proto, x3_proto=tir_proto)
                    else:
                        rgb_proto, nir_proto = model(x1_proto=rgb_proto, x2_proto=nir_proto)

                loss_p2t_1 = xent(rgb_proto, text_features1, target_uni, target_uni)
                loss_t2p_1 = xent(text_features1, rgb_proto, target_uni, target_uni)

                loss_p2t_2 = xent(nir_proto, text_features2, target_uni, target_uni)
                loss_t2p_2 = xent(text_features2, nir_proto, target_uni, target_uni)
                
                if cfg.DATASETS.NAMES != 'RGBN300':
                    loss_p2t_3 = xent(tir_proto, text_features3, target_uni, target_uni)
                    loss_t2p_3 = xent(text_features3, tir_proto, target_uni, target_uni)

                loss1, loss_item_list1 = loss_fn(mode1[0], mode1[1], target, target_cam)
                loss2, loss_item_list2 = loss_fn(mode2[0], mode2[1], target, target_cam)
                if cfg.DATASETS.NAMES != 'RGBN300':
                    loss3, loss_item_list3 = loss_fn(mode3[0], mode3[1], target, target_cam)

                # 3 image align to text prompt 
                batch = cfg.SOLVER.IMS_PER_BATCH
                num_classes = model.num_classes
                i_ter = num_classes // batch
                left = num_classes-batch* (num_classes//batch)
                if left != 0 :
                    i_ter = i_ter+1
                text_features1 = []
                text_features2 = []
                if cfg.DATASETS.NAMES != 'RGBN300':
                    text_features3 = []
                with torch.no_grad():
                    for i in range(i_ter):
                        if i+1 != i_ter:
                            l_list = torch.arange(i*batch, (i+1)* batch)
                        else:
                            l_list = torch.arange(i*batch, num_classes)
                        with amp.autocast(enabled=True):
                            if cfg.DATASETS.NAMES != 'RGBN300':
                                text_feature1, text_feature2, text_feature3 = model(label = l_list, get_text = True, force_traning=True)
                            else:
                                text_feature1, text_feature2 = model(label = l_list, get_text = True)
                        text_features1.append(text_feature1.cpu())
                        text_features2.append(text_feature2.cpu())
                        if cfg.DATASETS.NAMES != 'RGBN300':
                            text_features3.append(text_feature3.cpu())
                    text_features1 = torch.cat(text_features1, 0).cuda()
                    text_features2 = torch.cat(text_features2, 0).cuda()
                    if cfg.DATASETS.NAMES != 'RGBN300':
                        text_features3 = torch.cat(text_features3, 0).cuda()

                logits1 = mode1[2] @ text_features1.t()
                logits2 = mode2[2] @ text_features2.t()
                if cfg.DATASETS.NAMES != 'RGBN300':
                    logits3 = mode3[2] @ text_features3.t()

                loss_i2tce_1 = F.cross_entropy(logits1, target)
                loss_i2tce_2 = F.cross_entropy(logits2, target)
                if cfg.DATASETS.NAMES != 'RGBN300':
                    loss_i2tce_3 = F.cross_entropy(logits3, target)

                if 'RGB' == cfg.MODEL.MULTI_MODAL:
                    loss = loss1
                else:
                    if epoch > 20: 
                        aplha = 0.9
                        i2p_factor = cfg.MODEL.I2TCE_LOSS
                        t2p_factor = cfg.MODEL.T2P_LOSS
                        i2tce_factor = cfg.MODEL.I2P_LOSS
                    else:
                        aplha = 0.1
                        i2p_factor = cfg.MODEL.I2P_LOSS
                        t2p_factor = cfg.MODEL.T2P_LOSS
                        i2tce_factor = cfg.MODEL.I2TCE_LOSS
                        
                    if cfg.DATASETS.NAMES != 'RGBN300':
                        loss = 0.33*loss1+0.33*loss2+0.33*loss3 \
                            + i2p_factor*(0.33*loss_i2p_rgb + 0.33*loss_i2p_nir + 0.33*loss_i2p_tir) \
                            + t2p_factor*(0.33*(loss_p2t_1 + loss_t2p_1) + 0.33*(loss_p2t_2 + loss_t2p_2) + 0.33*(loss_p2t_3 + loss_t2p_3)) \
                            + i2tce_factor*(0.33*loss_i2tce_1 + 0.33*loss_i2tce_2 + 0.33*loss_i2tce_3)
                    else:
                        loss = 0.5*loss1+0.5*loss2 \
                            + (1-aplha)*(0.5*loss_i2p_rgb + 0.5*loss_i2p_nir) \
                            + 0.5*(loss_p2t_1 + loss_t2p_1) + 0.5*(loss_p2t_2 + loss_t2p_2) \
                            + aplha*(0.5*loss_i2tce_1 + 0.5*loss_i2tce_2)
                        
                if epoch % eval_period == 0:
                    if cfg.DATASETS.NAMES != 'RGBN300':
                        for i, img_feat1, img_feat2, img_feat3 in zip(target, mode1[2], mode2[2], mode3[2]):
                            labels.append(i)
                    else:
                        for i, img_feat1, img_feat2 in zip(target, mode1[2], mode2[2]):
                            labels.append(i)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()
            if isinstance(mode1[0][0], list):
                acc = (mode1[0][0].max(1)[1] == target).float().mean()
                acc1 = (mode2[0][0].max(1)[1] == target).float().mean()
                if cfg.DATASETS.NAMES != 'RGBN300':
                    acc2 = (mode3[0][0].max(1)[1] == target).float().mean()
            else:
                acc = (mode1[0][0].max(1)[1] == target).float().mean()
                acc1 = (mode2[0][0].max(1)[1] == target).float().mean()
                if cfg.DATASETS.NAMES != 'RGBN300':
                    acc2 = (mode3[0][0].max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img1.shape[0])
            loss_meter1.update(loss1.item(), img1.shape[0])
            loss_meter2.update(loss2.item(), img1.shape[0])
            loss_meter4.update(loss_i2p_rgb.item(), img1.shape[0])
            loss_meter5.update(loss_i2p_nir.item(), img1.shape[0])
            
            loss_meter7.update(loss_i2tce_1.item(), img1.shape[0])
            loss_meter8.update(loss_i2tce_2.item(), img1.shape[0])
                        
            loss_meter10.update(loss_p2t_1.item() + loss_t2p_1.item(), rgb_proto.shape[0])
            loss_meter11.update(loss_p2t_2.item() + loss_t2p_2.item(), rgb_proto.shape[0])
            

            acc_meter.update(acc, 1)
            acc_meter1.update(acc1, 1)
            
            if cfg.DATASETS.NAMES != 'RGBN300':
                loss_meter3.update(loss3.item(), img1.shape[0])
                loss_meter6.update(loss_i2p_tir.item(), img1.shape[0])
                loss_meter9.update(loss_i2tce_3.item(), img1.shape[0])
                loss_meter12.update(loss_p2t_3.item() + loss_t2p_3.item(), rgb_proto.shape[0])
                acc_meter2.update(acc2, 1)

            torch.cuda.synchronize()
    
            if (n_iter + 1) % log_period == 0 and local_rank < 1:
                lr_print = optimizer.state_dict()['param_groups'][0]['lr']
                if cfg.DATASETS.NAMES != 'RGBN300':
                    logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Loss_Branch: [{:.3f}, {:.3f}, {:.3f}], Loss_I2P: [{:.3f}, {:.3f}, {:.3f}], Loss_T2P: [{:.3f}, {:.3f}, {:.3f}], Loss_I2T: [{:.3f}, {:.3f}, {:.3f}], Acc: [{:.3f}, {:.3f}, {:.3f}], Base Lr: {:.2e}"
                                .format(epoch, (n_iter + 1), len(train_loader),
                                        loss_meter.avg, loss_meter1.avg, loss_meter2.avg, loss_meter3.avg, loss_meter4.avg, loss_meter5.avg, loss_meter6.avg, loss_meter10.avg, loss_meter11.avg, loss_meter12.avg, loss_meter7.avg, loss_meter8.avg, loss_meter9.avg, acc_meter.avg, acc_meter1.avg, acc_meter2.avg, lr_print))
                else:
                    logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Loss_Branch: [{:.3f}, {:.3f}], Loss_I2P: [{:.3f}, {:.3f}], Loss_T2P: [{:.3f}, {:.3f}], Loss_I2T: [{:.3f}, {:.3f}], Acc: [{:.3f}, {:.3f}], Base Lr: {:.2e}"
                                .format(epoch, (n_iter + 1), len(train_loader),
                                        loss_meter.avg, loss_meter1.avg, loss_meter2.avg, loss_meter4.avg, loss_meter5.avg, loss_meter10.avg, loss_meter11.avg, loss_meter7.avg, loss_meter8.avg, acc_meter.avg, acc_meter1.avg, lr_print))
        if epoch % eval_period == 0:
            batch = 64
            num_classes = model.num_classes
            i_ter = num_classes // batch
            left = num_classes-batch* (num_classes//batch)
            if left != 0 :
                i_ter = i_ter+1

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if local_rank < 1:
                    model.eval()
                    for n_iter, (img1, img2, img3, vid, camid, camids, sceneid, img_paths) in enumerate(val_loader):
                        with torch.no_grad():
                            img1 = img1.to(device)
                            img2 = img2.to(device)
                            if cfg.DATASETS.NAMES != 'RGBN300':
                                img3 = img3.to(device)
                            if cfg.MODEL.SIE_CAMERA:
                                camids = camids.to(device)
                            else: 
                                camids = None
                            if cfg.MODEL.SIE_VIEW:
                                target_view = target_view.to(device)
                            else: 
                                target_view = None
                            if cfg.DATASETS.NAMES != 'RGBN300':
                                feat = model(img1, img2, img3, label = target, cam_label=camids, view_label=target_view)
                            else:
                                feat = model(img1, img2, label = target, cam_label=camids, view_label=target_view)
                            if cfg.DATASETS.NAMES in ['MSVR310']:
                                evaluator.update((feat, vid, camid, sceneid, img_paths))
                            else:
                                evaluator.update((feat, vid, camid, img_paths))
                    cmc, mAP, all_AP, q_pids, _, _, _, _, _ = evaluator.compute(epoch)
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

                    logger_eval.info("Validation Results - Epoch: {}".format(epoch))
                    logger_eval.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger_eval.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

                    torch.cuda.empty_cache()
            else:
                model.eval()
                for n_iter, (img1, img2, img3, vid, camid, camids, sceneid, img_paths) in enumerate(val_loader):
                    with torch.no_grad():
                        img1 = img1.to(device)
                        img2 = img2.to(device)
                        if cfg.DATASETS.NAMES != 'RGBN300':
                            img3 = img3.to(device)
                        if cfg.MODEL.SIE_CAMERA:
                            camids = camids.to(device)
                        else: 
                            camids = None
                        if cfg.MODEL.SIE_VIEW:
                            target_view = target_view.to(device)
                        else: 
                            target_view = None
                        if cfg.DATASETS.NAMES != 'RGBN300':
                            feat, feat_bn = model(img1, img2, img3, label = target, cam_label=camids, view_label=target_view)
                        else:
                            feat, feat_bn = model(img1, img2, label = target, cam_label=camids, view_label=target_view)

                    eval_logger.update(feat, feat_bn, vid, camid, sceneid, img_paths)

                # eval_logger.log_evaluation(model, epoch, logger, logger_eval, logger_eval_bn)
                eval_logger.log_evaluation_all(model, epoch, logger, logger_eval, logger_eval_bn)

                torch.cuda.empty_cache()
		
    all_end_time = time.monotonic()
    total_time = timedelta(seconds=all_end_time - all_start_time)
    logger.info("Total running time: {}".format(total_time))
    print(cfg.OUTPUT_DIR)


def do_inference(cfg,
                 model,
                 val_loader,
                 cam_loader, 
                 num_query):
    device = "cuda"
    logger = logging.getLogger("ICPL.test")
    logger_eval = logging.getLogger("evaluator")
    logger_eval_bn = logging.getLogger("evaluator_bn")
    logger.info("Enter inferencing")

    eval_logger = PrettyEvaluation(cfg, num_query)
    eval_logger.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()

    for n_iter, (img1, img2, img3, vid, camid, camids, sceneid, img_paths) in enumerate(val_loader):
        with torch.no_grad():
            img1 = img1.to(device)
            img2 = img2.to(device)
            if cfg.DATASETS.NAMES != 'RGBN300':
                img3 = img3.to(device)
            target = None
            if cfg.MODEL.SIE_CAMERA:
                camids = camids.to(device)
            else: 
                camids = None
            if cfg.MODEL.SIE_VIEW:
                target_view = target_view.to(device)
            else: 
                target_view = None
            if cfg.DATASETS.NAMES != 'RGBN300':
                feat, feat_bn = model(img1, img2, img3, target, cam_label=camids, view_label=target_view)
            else:
                feat, feat_bn = model(img1, img2, label = target, cam_label=camids, view_label=target_view)

            eval_logger.update(feat, feat_bn, vid, camid, sceneid, img_paths)

        epoch = 1

    eval_logger.log_evaluation(model, epoch, logger, logger_eval, logger_eval_bn)
    # eval_logger.log_evaluation_all(model, epoch, logger, logger_eval, logger_eval_bn)

from utils.logger import setup_logger
from datasets.make_dataloader_multi_modal import make_dataloader
from model.make_model import make_model
from solver.make_optimizer import make_optimizer
from solver.lr_scheduler import WarmupMultiStepLR, WarmupCosineWithHardRestartsSchedule, WarmupCosineSchedule100, WarmupCosineSchedule
from loss.make_loss import make_loss
from processor.processor import do_train
import random
import torch
import numpy as np
import os
import argparse
from config import cfg
import time

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="configs/person/vit_base.yml", help="path to config file", type=str
    )

    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    set_seed(cfg.SOLVER.SEED)

    if cfg.MODEL.DIST_TRAIN:
        torch.cuda.set_device(args.local_rank)

    output_dir = cfg.OUTPUT_DIR
    datasets_name = cfg.DATASETS.NAMES
    date_str = time.strftime('%y-%m-%d_%H_%M_%S',time.localtime(time.time()))
    output_dir = os.path.join(output_dir, datasets_name, date_str)
    cfg.merge_from_list(['OUTPUT_DIR', output_dir])
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger_eval= setup_logger("evaluator", output_dir, if_train=True)
    logger_eval_bn= setup_logger("evaluator_bn", output_dir, if_train=True)
    logger = setup_logger("ICPL", output_dir, if_train=True)
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DIST_TRAIN:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    train_loader, train_loader_normal, val_loader, cam_loader, num_query, num_classes, camera_num, scene_num = make_dataloader(cfg)

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, scene_num = scene_num)

    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)

    optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)

    if cfg.SOLVER.SCHEDULER == 'step':
        scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR, cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)
    elif cfg.SOLVER.SCHEDULER == 'cosine100':
        scheduler = WarmupCosineSchedule100(optimizer, warmup_steps=cfg.SOLVER.WARMUP_ITERS, t_total=cfg.SOLVER.MAX_EPOCHS, cycles=cfg.SOLVER.COS_CYCLES)
    elif cfg.SOLVER.SCHEDULER == 'cosine':
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=cfg.SOLVER.WARMUP_ITERS, t_total=cfg.SOLVER.MAX_EPOCHS, cycles=cfg.SOLVER.COS_CYCLES)

    do_train(
        cfg,
        model,
        center_criterion,
        train_loader,
        train_loader_normal,
        val_loader,
        cam_loader,
        optimizer,
        optimizer_center,
        scheduler,
        loss_func,
        num_query, args.local_rank
    )

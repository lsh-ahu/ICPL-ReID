import logging
import os
import sys
import os.path as osp
from prettytable import PrettyTable
import torch
from utils.metrics import R1_mAP_eval_MSVR310
from utils.metrics import R1_mAP_eval

def setup_logger(name, save_dir, if_train):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    
    if 'evaluator' != name and 'evaluator_bn' != name:
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    if save_dir:
        if not osp.exists(save_dir):
            os.makedirs(save_dir)
        if 'evaluator' == name:
            fh = logging.FileHandler(os.path.join(save_dir, "eval_log.txt"), mode='w')
        elif 'evaluator_bn' == name:
            fh = logging.FileHandler(os.path.join(save_dir, "eval_bn_log.txt"), mode='w')
        else:
            if if_train:
                fh = logging.FileHandler(os.path.join(save_dir, "train_log.txt"), mode='w')
            else:
                fh = logging.FileHandler(os.path.join(save_dir, "test_log.txt"), mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

class PrettyEvaluation():
    def __init__(self, cfg, num_query):
        self.cfg = cfg
        self.best_index = {'mAP': 0, 'Epoch': 0, 'Rank-1': 0, 'Rank-5': 0, 'Rank-10': 0}
        self.best_bn_index = {'mAP': 0, 'Epoch': 0, 'Rank-1': 0, 'Rank-5': 0, 'Rank-10': 0}
        
        if cfg.DATASETS.NAMES in ['MSVR310', 'RGBN300', 'RGBNT100']:
            self.evaluator = R1_mAP_eval_MSVR310(cfg.DATASETS.NAMES, num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, cfg=cfg)
            self.evaluator_rgb = R1_mAP_eval_MSVR310(cfg.DATASETS.NAMES, num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, cfg=cfg)
            self.evaluator_nir = R1_mAP_eval_MSVR310(cfg.DATASETS.NAMES, num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, cfg=cfg)
            self.evaluator_tir = R1_mAP_eval_MSVR310(cfg.DATASETS.NAMES, num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, cfg=cfg)
            self.evaluator_rgb_nir = R1_mAP_eval_MSVR310(cfg.DATASETS.NAMES, num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, cfg=cfg)
            self.evaluator_nir_tir = R1_mAP_eval_MSVR310(cfg.DATASETS.NAMES, num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, cfg=cfg)
            self.evaluator_tir_rgb = R1_mAP_eval_MSVR310(cfg.DATASETS.NAMES, num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, cfg=cfg)
            self.evaluator_bn = R1_mAP_eval_MSVR310(cfg.DATASETS.NAMES, num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, cfg=cfg)
            self.evaluator_bn_rgb = R1_mAP_eval_MSVR310(cfg.DATASETS.NAMES, num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, cfg=cfg)
            self.evaluator_bn_nir = R1_mAP_eval_MSVR310(cfg.DATASETS.NAMES, num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, cfg=cfg)
            self.evaluator_bn_tir = R1_mAP_eval_MSVR310(cfg.DATASETS.NAMES, num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, cfg=cfg)
            self.evaluator_bn_rgb_nir = R1_mAP_eval_MSVR310(cfg.DATASETS.NAMES, num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, cfg=cfg)
            self.evaluator_bn_nir_tir = R1_mAP_eval_MSVR310(cfg.DATASETS.NAMES, num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, cfg=cfg)
            self.evaluator_bn_tir_rgb = R1_mAP_eval_MSVR310(cfg.DATASETS.NAMES, num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, cfg=cfg)
        else:
            self.evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, cfg=cfg)
            self.evaluator_rgb = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, cfg=cfg)
            self.evaluator_nir = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, cfg=cfg)
            self.evaluator_tir = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, cfg=cfg)
            self.evaluator_rgb_nir = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, cfg=cfg)
            self.evaluator_nir_tir = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, cfg=cfg)
            self.evaluator_tir_rgb = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, cfg=cfg)
            self.evaluator_bn = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, cfg=cfg)
            self.evaluator_bn_rgb = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, cfg=cfg)
            self.evaluator_bn_nir = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, cfg=cfg)
            self.evaluator_bn_tir = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, cfg=cfg)
            self.evaluator_bn_rgb_nir = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, cfg=cfg)
            self.evaluator_bn_nir_tir = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, cfg=cfg)
            self.evaluator_bn_tir_rgb = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, cfg=cfg)

    def reset(self):
        self.evaluator.reset()
        self.evaluator_rgb.reset()
        self.evaluator_nir.reset()
        self.evaluator_tir.reset()
        self.evaluator_rgb_nir.reset()
        self.evaluator_nir_tir.reset()
        self.evaluator_tir_rgb.reset()
        self.evaluator_bn.reset()
        self.evaluator_bn_rgb.reset()
        self.evaluator_bn_nir.reset()
        self.evaluator_bn_tir.reset()
        self.evaluator_bn_rgb_nir.reset()
        self.evaluator_bn_nir_tir.reset()
        self.evaluator_bn_tir_rgb.reset()

    def update(self, feat, feat_bn, vid, camid, sceneid, img_paths):
        feat_mm = torch.chunk(feat, 3, dim=1)
        feat_bn_mm = torch.chunk(feat_bn, 3, dim=1)

        feat_rgb_nir = torch.cat([feat_mm[0], feat_mm[1]], dim=1)
        feat_nir_tir = torch.cat([feat_mm[1], feat_mm[2]], dim=1)
        feat_tir_rgb = torch.cat([feat_mm[2], feat_mm[0]], dim=1)

        feat_bn_rgb_nir = torch.cat([feat_bn_mm[0], feat_bn_mm[1]], dim=1)
        feat_bn_nir_tir = torch.cat([feat_bn_mm[1], feat_bn_mm[2]], dim=1)
        feat_bn_tir_rgb = torch.cat([feat_bn_mm[2], feat_bn_mm[0]], dim=1)

        if self.cfg.DATASETS.NAMES in ['MSVR310', 'RGBN300', 'RGBNT100']:
            self.evaluator.update((feat, vid, camid, sceneid, img_paths))
            self.evaluator_rgb.update((feat_mm[0], vid, camid, sceneid, img_paths))
            self.evaluator_nir.update((feat_mm[1], vid, camid, sceneid, img_paths))
            if self.cfg.DATASETS.NAMES != 'RGBN300':
                self.evaluator_tir.update((feat_mm[2], vid, camid, sceneid, img_paths))

                self.evaluator_rgb_nir.update((feat_rgb_nir, vid, camid, sceneid, img_paths))
                self.evaluator_nir_tir.update((feat_nir_tir, vid, camid, sceneid, img_paths))
                self.evaluator_tir_rgb.update((feat_tir_rgb, vid, camid, sceneid, img_paths))

            self.evaluator_bn.update((feat_bn, vid, camid, sceneid, img_paths))
            self.evaluator_bn_rgb.update((feat_bn_mm[0], vid, camid, sceneid, img_paths))
            self.evaluator_bn_nir.update((feat_bn_mm[1], vid, camid, sceneid, img_paths))
            if self.cfg.DATASETS.NAMES != 'RGBN300':
                self.evaluator_bn_tir.update((feat_bn_mm[2], vid, camid, sceneid, img_paths))
                
                self.evaluator_bn_rgb_nir.update((feat_bn_rgb_nir, vid, camid, sceneid, img_paths))
                self.evaluator_bn_nir_tir.update((feat_bn_nir_tir, vid, camid, sceneid, img_paths))
                self.evaluator_bn_tir_rgb.update((feat_bn_tir_rgb, vid, camid, sceneid, img_paths))
        else:
            self.evaluator.update((feat, vid, camid, img_paths))
            self.evaluator_rgb.update((feat_mm[0], vid, camid, img_paths))
            self.evaluator_nir.update((feat_mm[1], vid, camid, img_paths))
            self.evaluator_tir.update((feat_mm[2], vid, camid, img_paths))

            self.evaluator_rgb_nir.update((feat_rgb_nir, vid, camid, img_paths))
            self.evaluator_nir_tir.update((feat_nir_tir, vid, camid, img_paths))
            self.evaluator_tir_rgb.update((feat_tir_rgb, vid, camid, img_paths))

            self.evaluator_bn.update((feat_bn, vid, camid, img_paths))
            self.evaluator_bn_rgb.update((feat_bn_mm[0], vid, camid, img_paths))
            self.evaluator_bn_nir.update((feat_bn_mm[1], vid, camid, img_paths))
            self.evaluator_bn_tir.update((feat_bn_mm[2], vid, camid, img_paths))

            self.evaluator_bn_rgb_nir.update((feat_bn_rgb_nir, vid, camid, img_paths))
            self.evaluator_bn_nir_tir.update((feat_bn_nir_tir, vid, camid, img_paths))
            self.evaluator_bn_tir_rgb.update((feat_bn_tir_rgb, vid, camid, img_paths))

    def log_evaluation_all(self, model, epoch, logger, logger_eval, logger_eval_bn):
        table = PrettyTable([f"Epoch:{epoch}", "mAP", "R-1", "R-5", "R-10"])

        # MM 
        cmc, mAP, all_AP, q_pids, _, _, _, _, _ = self.evaluator.compute(epoch)
        table.add_row(['MM', mAP, cmc[0], cmc[4], cmc[9]])
        if mAP >= self.best_index['mAP']:
            self.best_index['mAP'] = mAP
            self.best_index['Epoch'] = epoch
            self.best_index['Rank-1'] = cmc[0]
            self.best_index['Rank-5'] = cmc[4]
            self.best_index['Rank-10'] = cmc[9]
            if self.cfg.SOLVER.SAVE_CHECKPOINT:
                torch.save(model.state_dict(),
                            os.path.join(self.cfg.OUTPUT_DIR, self.cfg.MODEL.NAME + '-best.pth'))
            
        cmc, mAP, all_AP, q_pids, _, _, _, _, _ = self.evaluator_rgb.compute(epoch)
        table.add_row(['RGB', mAP, cmc[0], cmc[4], cmc[9]])
        cmc, mAP, all_AP, q_pids, _, _, _, _, _ = self.evaluator_nir.compute(epoch)
        table.add_row(['NIR', mAP, cmc[0], cmc[4], cmc[9]])
        cmc, mAP, all_AP, q_pids, _, _, _, _, _ = self.evaluator_tir.compute(epoch)
        table.add_row(['TIR', mAP, cmc[0], cmc[4], cmc[9]])

        cmc, mAP, all_AP, q_pids, _, _, _, _, _ = self.evaluator_rgb_nir.compute(epoch)
        table.add_row(['RGB_NIR', mAP, cmc[0], cmc[4], cmc[9]])
        cmc, mAP, all_AP, q_pids, _, _, _, _, _ = self.evaluator_nir_tir.compute(epoch)
        table.add_row(['NIR_TIR', mAP, cmc[0], cmc[4], cmc[9]])
        cmc, mAP, all_AP, q_pids, _, _, _, _, _ = self.evaluator_tir_rgb.compute(epoch)
        table.add_row(['TIR_RGB', mAP, cmc[0], cmc[4], cmc[9]])
        table.add_row([f'Best:{self.best_index["Epoch"]}', self.best_index['mAP'], self.best_index['Rank-1'], self.best_index['Rank-5'], self.best_index['Rank-10']])
        
        table.add_row(["-"*6, "-"*5, "-"*5, "-"*5, "-"*5])
        ######################################################################
        cmc, mAP, all_AP, q_pids, _, _, _, _, _ = self.evaluator_bn.compute(epoch)
        table.add_row(['MM_BN', mAP, cmc[0], cmc[4], cmc[9]])
        if mAP >= self.best_bn_index['mAP']:
            self.best_bn_index['mAP'] = mAP
            self.best_bn_index['Epoch'] = epoch
            self.best_bn_index['Rank-1'] = cmc[0]
            self.best_bn_index['Rank-5'] = cmc[4]
            self.best_bn_index['Rank-10'] = cmc[9]
            if self.cfg.SOLVER.SAVE_CHECKPOINT:
                torch.save(model.state_dict(),
                            os.path.join(self.cfg.OUTPUT_DIR, self.cfg.MODEL.NAME + '-best_bn.pth'))
            
        cmc, mAP, all_AP, q_pids, _, _, _, _, _ = self.evaluator_bn_rgb.compute(epoch)
        table.add_row(['RGB', mAP, cmc[0], cmc[4], cmc[9]])
        cmc, mAP, all_AP, q_pids, _, _, _, _, _ = self.evaluator_bn_nir.compute(epoch)
        table.add_row(['NIR', mAP, cmc[0], cmc[4], cmc[9]])
        cmc, mAP, all_AP, q_pids, _, _, _, _, _ = self.evaluator_bn_tir.compute(epoch)
        table.add_row(['TIR', mAP, cmc[0], cmc[4], cmc[9]])

        cmc, mAP, all_AP, q_pids, _, _, _, _, _ = self.evaluator_bn_rgb_nir.compute(epoch)
        table.add_row(['RGB_NIR', mAP, cmc[0], cmc[4], cmc[9]])
        cmc, mAP, all_AP, q_pids, _, _, _, _, _ = self.evaluator_bn_nir_tir.compute(epoch)
        table.add_row(['NIR_TIR', mAP, cmc[0], cmc[4], cmc[9]])
        cmc, mAP, all_AP, q_pids, _, _, _, _, _ = self.evaluator_bn_tir_rgb.compute(epoch)
        table.add_row(['TIR_RGB', mAP, cmc[0], cmc[4], cmc[9]])
        table.add_row([f'Best:{self.best_bn_index["Epoch"]}', self.best_bn_index['mAP'], self.best_bn_index['Rank-1'], self.best_bn_index['Rank-5'], self.best_bn_index['Rank-10']])

        table.custom_format["mAP"] = lambda f, v: f"{v:.1%}" if v != '-----' else '-----'
        table.custom_format["R-1"] = lambda f, v: f"{v:.1%}" if v != '-----' else '-----'
        table.custom_format["R-5"] = lambda f, v: f"{v:.1%}" if v != '-----' else '-----'
        table.custom_format["R-10"] = lambda f, v: f"{v:.1%}" if v != '-----' else '-----'
        table.align = "c"
        logger.info('\n' + str(table))

        sub_table = PrettyTable()
        sub_table.field_names = table.field_names
        sub_table.add_rows(table.rows[:8])
        sub_table.custom_format["mAP"] = lambda f, v: f"{v:.1%}" if v != '-----' else '-----'
        sub_table.custom_format["R-1"] = lambda f, v: f"{v:.1%}" if v != '-----' else '-----'
        sub_table.custom_format["R-5"] = lambda f, v: f"{v:.1%}" if v != '-----' else '-----'
        sub_table.custom_format["R-10"] = lambda f, v: f"{v:.1%}" if v != '-----' else '-----'
        sub_table.align = "c"
        logger_eval.info('\n' + str(sub_table))
        logger_eval.info(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        
        sub_table = PrettyTable()
        sub_table.field_names = table.field_names
        sub_table.add_rows(table.rows[-8:])
        sub_table.custom_format["mAP"] = lambda f, v: f"{v:.1%}" if v != '-----' else '-----'
        sub_table.custom_format["R-1"] = lambda f, v: f"{v:.1%}" if v != '-----' else '-----'
        sub_table.custom_format["R-5"] = lambda f, v: f"{v:.1%}" if v != '-----' else '-----'
        sub_table.custom_format["R-10"] = lambda f, v: f"{v:.1%}" if v != '-----' else '-----'
        sub_table.align = "c"
        logger_eval_bn.info('\n' + str(sub_table))
        logger_eval_bn.info(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

    def log_evaluation(self, model, epoch, logger, logger_eval, logger_eval_bn):
        table = PrettyTable([f"Epoch:{epoch}", "mAP", "R-1", "R-5", "R-10"])

        # MM 
        cmc, mAP, all_AP, q_pids, _, _, _, _, _ = self.evaluator.compute(epoch)
        table.add_row(['MM', mAP, cmc[0], cmc[4], cmc[9]])
        if mAP >= self.best_index['mAP']:
            self.best_index['mAP'] = mAP
            self.best_index['Epoch'] = epoch
            self.best_index['Rank-1'] = cmc[0]
            self.best_index['Rank-5'] = cmc[4]
            self.best_index['Rank-10'] = cmc[9]
            if self.cfg.SOLVER.SAVE_CHECKPOINT:
                torch.save(model.state_dict(),
                            os.path.join(self.cfg.OUTPUT_DIR, self.cfg.MODEL.NAME + '-best.pth'))

        table.add_row([f'Best:{self.best_index["Epoch"]}', self.best_index['mAP'], self.best_index['Rank-1'], self.best_index['Rank-5'], self.best_index['Rank-10']])
        
        table.add_row(["-"*6, "-"*5, "-"*5, "-"*5, "-"*5])
        ######################################################################
        cmc, mAP, all_AP, q_pids, _, _, _, _, _ = self.evaluator_bn.compute(epoch)
        table.add_row(['MM_BN', mAP, cmc[0], cmc[4], cmc[9]])
        if mAP >= self.best_bn_index['mAP']:
            self.best_bn_index['mAP'] = mAP
            self.best_bn_index['Epoch'] = epoch
            self.best_bn_index['Rank-1'] = cmc[0]
            self.best_bn_index['Rank-5'] = cmc[4]
            self.best_bn_index['Rank-10'] = cmc[9]
            if self.cfg.SOLVER.SAVE_CHECKPOINT:
                torch.save(model.state_dict(),
                            os.path.join(self.cfg.OUTPUT_DIR, self.cfg.MODEL.NAME + '-best_bn.pth'))
        table.add_row([f'Best:{self.best_bn_index["Epoch"]}', self.best_bn_index['mAP'], self.best_bn_index['Rank-1'], self.best_bn_index['Rank-5'], self.best_bn_index['Rank-10']])

        table.custom_format["mAP"] = lambda f, v: f"{v:.1%}" if v != '-----' else '-----'
        table.custom_format["R-1"] = lambda f, v: f"{v:.1%}" if v != '-----' else '-----'
        table.custom_format["R-5"] = lambda f, v: f"{v:.1%}" if v != '-----' else '-----'
        table.custom_format["R-10"] = lambda f, v: f"{v:.1%}" if v != '-----' else '-----'
        table.align = "c"
        logger.info('\n' + str(table))

        sub_table = PrettyTable()
        sub_table.field_names = table.field_names
        sub_table.add_rows(table.rows[:2])
        sub_table.custom_format["mAP"] = lambda f, v: f"{v:.1%}" if v != '-----' else '-----'
        sub_table.custom_format["R-1"] = lambda f, v: f"{v:.1%}" if v != '-----' else '-----'
        sub_table.custom_format["R-5"] = lambda f, v: f"{v:.1%}" if v != '-----' else '-----'
        sub_table.custom_format["R-10"] = lambda f, v: f"{v:.1%}" if v != '-----' else '-----'
        sub_table.align = "c"
        logger_eval.info('\n' + str(sub_table))
        logger_eval.info(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        
        sub_table = PrettyTable()
        sub_table.field_names = table.field_names
        sub_table.add_rows(table.rows[-2:])
        sub_table.custom_format["mAP"] = lambda f, v: f"{v:.1%}" if v != '-----' else '-----'
        sub_table.custom_format["R-1"] = lambda f, v: f"{v:.1%}" if v != '-----' else '-----'
        sub_table.custom_format["R-5"] = lambda f, v: f"{v:.1%}" if v != '-----' else '-----'
        sub_table.custom_format["R-10"] = lambda f, v: f"{v:.1%}" if v != '-----' else '-----'
        sub_table.align = "c"
        logger_eval_bn.info('\n' + str(sub_table))
        logger_eval_bn.info(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

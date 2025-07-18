import os
from config import cfg
import argparse
from datasets.make_dataloader_multi_modal import make_dataloader
from model.make_model import make_model
from processor.processor import do_inference
from utils.logger import setup_logger
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="configs/person/vit_base.yml", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    datasets_name = cfg.DATASETS.NAMES
    date_str = time.strftime('%y-%m-%d_%H_%M_%S',time.localtime(time.time()))
    output_dir = os.path.join(output_dir, datasets_name, date_str)
    cfg.merge_from_list(['OUTPUT_DIR', output_dir])
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger_eval= setup_logger("evaluator", output_dir, if_train=True)
    logger_eval_bn= setup_logger("evaluator_bn", output_dir, if_train=True)
    logger = setup_logger("ICPL", output_dir, if_train=False)
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    train_loader, train_loader_normal, val_loader, cam_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, scene_num = view_num)

    model.load_param(cfg.TEST.WEIGHT)

    do_inference(cfg, model, val_loader, cam_loader, num_query)
    logger.info(output_dir)



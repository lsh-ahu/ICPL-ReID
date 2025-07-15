import torch
import logging

def make_optimizer(cfg, model, center_criterion):
    logger = logging.getLogger("ICPL.image_train")
    logger.info('Turning off gradients in both the image and the text encoder')

    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad or 'text_encoder' in key:
            value.requires_grad_(False)
            continue

        if cfg.MODEL.MLP_ADAPTER:
            freeze_keywords = [
                'prompt_learner', 'adapter', 'Adapter', 'classifier', 'image_encoder.proj', 'ln_post', 'cv_embed', 'bottleneck'
            ]
            if not any(kw in key for kw in freeze_keywords):
                value.requires_grad_(False)
                continue

        logger.info(key)
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if 'prompt_learner' in key:
            lr = cfg.SOLVER.PROMPT_LR
            weight_decay = cfg.SOLVER.PROMPT_WEIGHT_DECAY_BIAS
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        if cfg.SOLVER.LARGE_FC_LR:
            if "classifier" in key or "arcface" in key:
                lr = cfg.SOLVER.BASE_LR * 2
                print('Using two times learning rate for fc ')

        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.MOMENTUM)
    elif cfg.SOLVER.OPTIMIZER_NAME == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    else:
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params)
    optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=cfg.SOLVER.CENTER_LR)

    return optimizer, optimizer_center
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from .bases_multi_modal import ImageDataset
from timm.data.random_erasing import RandomErasing
from .sampler_multi_modal import RandomIdentitySampler
from .sampler_ddp_multi_modal import RandomIdentitySampler_DDP
import torch.distributed as dist
from .MSVR310 import MSVR310
from .WMVEID863 import WMVEID863
from .RGBNT201 import RGBNT201
from .Market1501_RGBNT import Market1501_RGBNT
from .RGBNT100 import RGBNT100
from .RGBN300 import RGBN300

__factory = {
    'MSVR310': MSVR310,
    'WMVEID863': WMVEID863,
    'RGBNT201': RGBNT201,
    'MARKET_RGBNT': Market1501_RGBNT,
    'RGBN300': RGBN300,
    'RGBNT100': RGBNT100,
}

def train_collate_fn(batch):
    img1, img2, img3, pids, camids, sceneids, _ = zip(*batch)
    
    pids = torch.tensor(pids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    sceneids = torch.tensor(sceneids, dtype=torch.int64)

    img1 = torch.stack(img1, dim=0)
    img2 = torch.stack(img2, dim=0)
    
    if img3[0] is not None:
        img3 = torch.stack(img3, dim=0)
    else:
        img3 = None

    return img1, img2, img3, pids, camids, sceneids

def val_collate_fn(batch):
    img1, img2, img3, pids, camids, sceneids, img_paths = zip(*batch)
    
    camids_batch = torch.tensor(camids, dtype=torch.int64)

    img1 = torch.stack(img1, dim=0)
    img2 = torch.stack(img2, dim=0)

    if img3[0] is not None:
        img3 = torch.stack(img3, dim=0)
    else:
        img3 = None

    return img1, img2, img3, pids, camids, camids_batch, sceneids, img_paths

def make_dataloader(cfg):
    random_erase = cfg.INPUT.RANDOM_ERASE
    if random_erase:
        train_transforms = T.Compose([
                T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
                T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
                T.Pad(cfg.INPUT.PADDING),
                T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
                T.ToTensor(),
                T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
                RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
            ])
    else:
        train_transforms = T.Compose([
                T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
                T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
                T.Pad(cfg.INPUT.PADDING),
                T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
                T.ToTensor(),
                T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            ])

    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    cam_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    num_workers = cfg.DATALOADER.NUM_WORKERS

    dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)


    train_set = ImageDataset(dataset.train, train_transforms, cfg.DATASETS.NAMES)
    train_set_normal = ImageDataset(dataset.train, val_transforms, cfg.DATASETS.NAMES)
    num_classes = dataset.num_train_pids
    cam_num = dataset.num_train_cams
    scene_num = dataset.num_train_scenes

    if 'triplet' in cfg.DATALOADER.SAMPLER:
        if cfg.MODEL.DIST_TRAIN:
            print('DIST_TRAIN START')
            mini_batch_size = cfg.SOLVER.IMS_PER_BATCH // dist.get_world_size()
            data_sampler = RandomIdentitySampler_DDP(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE)
            batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, mini_batch_size, True)
            train_loader = torch.utils.data.DataLoader(
                train_set,
                num_workers=num_workers,
                batch_sampler=batch_sampler,
                collate_fn=train_collate_fn,
                pin_memory=True,
            )
        else:
            train_loader = DataLoader(
                train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
                sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
                num_workers=num_workers, collate_fn=train_collate_fn
            )
    elif cfg.DATALOADER.SAMPLER == 'softmax2':
        print('using softmax sampler2')
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
            sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
            num_workers=num_workers, collate_fn=train_collate_fn
        )
    elif cfg.DATALOADER.SAMPLER == 'softmax':
        print('using softmax sampler')
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
            collate_fn=train_collate_fn
        )
    else:
        print('unsupported sampler! expected softmax or triplet but got {}'.format(cfg.SAMPLER))
    
    
    val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms, cfg.DATASETS.NAMES)
    
    cam_set = ImageDataset(dataset.query + dataset.gallery, cam_transforms, cfg.DATASETS.NAMES)

    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    train_loader_normal = DataLoader(
        train_set_normal, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=train_collate_fn
    )
    cam_loader = DataLoader(
        cam_set, batch_size=1, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    return train_loader, train_loader_normal, val_loader, cam_loader, len(dataset.query), num_classes, cam_num, scene_num

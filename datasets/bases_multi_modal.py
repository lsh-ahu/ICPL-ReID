import sys

from PIL import Image, ImageFile

from torch.utils.data import Dataset
import os.path as osp
import random
import torch
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy
from einops import repeat
import numpy as np
import random

def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')  # 转换为灰度图像
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img

def crop_and_read_image(img_path, img_num=3):
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    
    cropped_images = []

    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')

            if img_num >= 1:
                img1 = img.crop((0, 0, 256, 128))
                cropped_images.append(img1)
            if img_num >= 2:
                img2 = img.crop((256, 0, 512, 128))
                cropped_images.append(img2)
            if img_num >= 3:
                img3 = img.crop((512, 0, 768, 128))
                cropped_images.append(img3)

            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass

    return tuple(cropped_images)

class BaseDataset(object):
    """
    Base class of reid dataset
    """

    def get_imagedata_info(self, data):
        pids, cams, scenes =[],[],[]

        for _,_,_, pid, sceneid, camid in data:
            pids += [pid]
            cams += [camid]
            scenes += [sceneid]

        pids = set(pids)
        cams = set(cams)
        scenes = set(scenes)
        
        num_pids = len(pids)
        num_cams = len(cams)
        num_scenes = len(scenes)
        num_imgs = len(data)

        return num_pids, num_imgs, num_cams, num_scenes

    def print_dataset_statistics(self):
        raise NotImplementedError

class BaseImageDataset(BaseDataset):
    """
    Base class of image reid dataset
    """

    def print_dataset_statistics(self, train, query, gallery):
        num_train_pids, num_train_imgs, num_train_cams, num_train_scenes = self.get_imagedata_info(train)
        num_query_pids, num_query_imgs, num_query_cams, num_query_scenes = self.get_imagedata_info(query)
        num_gallery_pids, num_gallery_imgs, num_gallery_cams, num_gallery_scenes = self.get_imagedata_info(gallery)

        print("Dataset statistics:")
        print("  ----------------------------------------------------")
        print("  subset   | # ids | # images | # cameras | # scenes ")
        print("  ----------------------------------------------------")
        print("  train    | {:5d} | {:8d} | {:9d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_cams, num_train_scenes))
        print("  query    | {:5d} | {:8d} | {:9d} | {:9d}".format(num_query_pids, num_query_imgs, num_query_cams, num_query_scenes))
        print("  gallery  | {:5d} | {:8d} | {:9d} | {:9d}".format(num_gallery_pids, num_gallery_imgs, num_gallery_cams, num_gallery_scenes))
        print("  ----------------------------------------------------")

class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None, dataset_names = None):
        self.dataset = dataset
        self.transform = transform
        self.use_crop_and_read = 'RGBNT100' == dataset_names
        self.use_crop_and_read_ni = 'RGBN300' == dataset_names

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path_rgb,img_path_ni,img_path_ti, pid, senceid, camid, = self.dataset[index]
        
        img_ti = None

        if self.use_crop_and_read:
            img_rgb, img_ni, img_ti = crop_and_read_image(img_path_rgb, img_num=3)
        elif self.use_crop_and_read_ni:
            img_rgb, img_ni = crop_and_read_image(img_path_rgb, img_num=2)
        else:
            img_rgb = read_image(img_path_rgb)
            img_ni = read_image(img_path_ni)
            img_ti = read_image(img_path_ti)

        img_rgb=numpy.array(img_rgb)
        img_ni=numpy.array(img_ni)

        img_rgb = Image.fromarray(img_rgb)
        img_ni = Image.fromarray(img_ni)

        if self.transform is not None:
            img_rgb = self.transform(img_rgb)
            img_ni = self.transform(img_ni)

        if img_ti is not None:
            img_ti = numpy.array(img_ti)
            img_ti = Image.fromarray(img_ti)
            if self.transform is not None:
                img_ti = self.transform(img_ti)

        return img_rgb, img_ni, img_ti, pid, camid, senceid, img_path_rgb
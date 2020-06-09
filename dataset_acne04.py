"""

"""
from torch.utils.data.dataset import Dataset

import random
import cv2
import sys
import numpy as np
import pandas as pd
import os
# import matplotlib.pyplot as plt
from easydict import EasyDict as ED

from .dataset import image_data_augmentation, Yolo_dataset
from cfg import Cfg


Cfg.dataset_dir = '/mnt/wenhao71/data/acne_article_data/filtered_images/'

label_map_dict = ED({
    'fore': 1,
})


def ACNE04(Yolo_dataset):
    """
    """
    def __init__(self, lable_path, cfg):
        """
        unlike in Yolo_dataset where the labels are stored in a txt file,
        with each line cls,x_center,y_center,w,h,
        annotations of ACNE04 are already converted into a csv file
        """
        if cfg.mixup == 2:
            raise ValueError("cutmix=1 - isn't supported for Detector")
        elif cfg.mixup == 2 and cfg.letter_box:
            raise ValueError("Combination: letter_box=1 & mosaic=1 - isn't supported, use only 1 of these parameters")

        self.cfg = cfg

        df_ann = pd.read_csv(lable_path)
        df_ann = df_ann[df_ann['class'].isin(label_map_dict.keys())].reset_index(drop=True)

        df_ann['xcen'] = df_ann.apply(lambda row: (row['xmax']+row['xmin'])/2/row['width'], axis=1)
        df_ann['ycen'] = df_ann.apply(lambda row: (row['ymax']+row['ymin'])/2/row['height'], axis=1)
        df_ann['box_width'] = df_ann['box_width'] / df_ann['width']
        df_ann['box_height'] = df_ann['box_height'] / df_ann['height']
        df_ann['class_index'] = df_ann['class'].apply(lambda c: label_map_dict[c])

        # each item of `truth` is of the form
        # key: filename of the image
        # (? to check)value: list of annotations of yolo format [class_index, xcen, ycen, w, h]
        truth = {k: [] for k in df_ann['filename'].tolist()}
        for _, row in df_ann.iterrows():
            truth[row['filename']].append(row[['class_index', 'xcen', 'ycen', 'box_width', 'box_height']].tolist())

        # f = open(lable_path, 'r', encoding='utf-8')
        # for line in f.readlines():
        #     data = line.split(" ")
        #     truth[data[0]] = []
        #     for i in data[1:]:
        #         truth[data[0]].append([int(j) for j in i.split(',')])

        self.truth = truth

        def __len__(self):
            return super().__len__()

        def __getitem__(self, index):
            return super().__getitem__(index)

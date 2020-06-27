"""

"""
import random
import sys
import os
from typing import Union, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from easydict import EasyDict as ED
import torch
from torchvision.transforms import functional as F
from torch.utils.data.dataset import Dataset

from dataset import image_data_augmentation, Yolo_dataset
from cfg_acne04 import Cfg


Cfg.dataset_dir = '/mnt/wenhao71/data/acne04/filtered_images/'

label_map_dict = ED({
    'acne': 0,
})


class ACNE04(Yolo_dataset):
    """
    """
    def __init__(self, label_path, cfg, train):
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
        self.train = train

        df_ann = pd.read_csv(label_path)
        df_ann = df_ann[df_ann['class'].isin(label_map_dict.keys())].reset_index(drop=True)

        # NOTE that the annotations used in this project are NOT in Yolo format, but in VOC format
        # ref Use_yolov4_to_train_your_own_data.md

        # df_ann['xcen'] = df_ann.apply(lambda row: (row['xmax']+row['xmin'])/2/row['width'], axis=1)
        # df_ann['ycen'] = df_ann.apply(lambda row: (row['ymax']+row['ymin'])/2/row['height'], axis=1)
        # df_ann['box_width'] = df_ann['box_width'] / df_ann['width']
        # df_ann['box_height'] = df_ann['box_height'] / df_ann['height']
        df_ann['class_index'] = df_ann['class'].apply(lambda c: label_map_dict[c])

        # each item of `truth` is of the form
        # key: filename of the image
        # value: list of annotations in the format [xmin, ymin, xmax, ymax, class_index]
        # truth = {k: [] for k in df_ann['filename'].tolist()}
        truth = {}
        for fn, df in df_ann.groupby("filename"):
            truth[fn] = df[['xmin', 'ymin', 'xmax', 'ymax', 'class_index']].values.astype(int).tolist()
        # for _, row in df_ann.iterrows():
        #     truth[row['filename']].append(row[['xmin', 'ymin', 'xmax', 'ymax', 'class_index']].tolist())

        # f = open(label_path, 'r', encoding='utf-8')
        # for line in f.readlines():
        #     data = line.split(" ")
        #     truth[data[0]] = []
        #     for i in data[1:]:
        #         truth[data[0]].append([int(j) for j in i.split(',')])

        self.truth = truth
        self.imgs = list(self.truth.keys())

    def __len__(self):
        return super().__len__()

    def __getitem__(self, index):
        if self.train:
            return super().__getitem__(index)
        else:
            return self._get_val_item(index)

    def _get_val_item(self, index):
        img_path = self.imgs[index]
        bboxes_with_cls_id = np.array(self.truth.get(img_path), dtype=np.float)
        img = cv2.imread(os.path.join(self.cfg.dataset_dir, img_path))
        img_height, img_width = img.shape[:2]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.cfg.w, self.cfg.h))
        # img = torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
        num_objs = len(bboxes_with_cls_id)
        target = {}
        # boxes should be in coco format
        boxes = bboxes_with_cls_id[...,:4]
        boxes[...,2] = (boxes[...,2] - boxes[...,0]) / img_width
        boxes[...,3] = (boxes[...,3] - boxes[...,1]) / img_height
        boxes[...,0] = boxes[...,0] / img_width
        boxes[...,1] = boxes[...,1] / img_height
        target['boxes'] = torch.as_tensor(boxes, dtype=torch.float32)
        target['labels'] = torch.as_tensor(bboxes_with_cls_id[...,-1].flatten(), dtype=torch.int64)
        target['image_id'] = torch.tensor([get_image_id(img_path)])
        target['area'] = (target['boxes'][:,3]-target['boxes'][:,1])*(target['boxes'][:,2]-target['boxes'][:,0])
        target['iscrowd'] = torch.zeros((num_objs,), dtype=torch.int64)
        return img, target


def train_val_test_split(df:pd.DataFrame, train_ratio:Union[int,float]=70, val_ratio:Union[int,float]=15, test_ratio:Union[int,float]=15) -> Tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    """
    """
    from random import shuffle
    from functools import reduce

    if isinstance(train_ratio, int):
        train_ratio = train_ratio / 100
        val_ratio = val_ratio / 100
        test_ratio = test_ratio / 100
    assert train_ratio+val_ratio+test_ratio == 1.0

    all_files_by_level = {f"lv{i}": [] for i in range(4)}
    for fn in df['filename'].unique():
        all_files_by_level[f"lv{fn.split('_')[0][-1]}"].append(fn)
    
    train, val, test = ({f"lv{i}": [] for i in range(4)} for _ in range(3))
    for i in range(4):
        shuffle(all_files_by_level[f"lv{i}"])
        lv_nb = len(all_files_by_level[f"lv{i}"])
        train[f"lv{i}"] = all_files_by_level[f"lv{i}"][:int(train_ratio*lv_nb)]
        val[f"lv{i}"] = all_files_by_level[f"lv{i}"][int(train_ratio*lv_nb): int((train_ratio+val_ratio)*lv_nb)]
        test[f"lv{i}"] = all_files_by_level[f"lv{i}"][int((train_ratio+val_ratio)*lv_nb):]
    train = reduce(lambda a,b: a+b, [v for _,v in train.items()])
    val = reduce(lambda a,b: a+b, [v for _,v in val.items()])
    test = reduce(lambda a,b: a+b, [v for _,v in test.items()])

    for i in range(4):
        print(f"lv{i}  ----- ", len(all_files_by_level[f"lv{i}"]))

    df_train = df[df['filename'].isin(train)].reset_index(drop=True)
    df_val = df[df['filename'].isin(val)].reset_index(drop=True)
    df_test = df[df['filename'].isin(test)].reset_index(drop=True)

    return df_train, df_val, df_test


def get_image_id(filename:str) -> int:
    """Convert a string to a integer."""
    lv, no = os.path.splitext(os.path.basename(filename))[0].split("_")
    lv = lv.replace("levle", "")
    no = f"{int(no):04d}"
    return int(lv+no)

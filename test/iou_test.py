# -*- coding: utf-8 -*-
'''

'''
import torch
import os, sys
from torch.nn import functional as F

import numpy as np


def bboxes_iou_test(bboxes_a, bboxes_b, fmt='voc', iou_type='iou'):
    """
    test function for bboxes_iou, adding message print and plot
    """
    if 'plt' not in dir():
        import matplotlib.pyplot as plt
    
    assert iou_type.lower() in ['iou', 'giou', 'diou', 'ciou']

    if isinstance(bboxes_a, np.ndarray):
        bboxes_a = torch.Tensor(bboxes_a)
    if isinstance(bboxes_b, np.ndarray):
        bboxes_b = torch.Tensor(bboxes_a)
    
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    N, K = bboxes_a.shape[0], bboxes_b.shape[0]
    # if N, K all equal 1, then plot

    # top left
    if fmt.lower() == 'voc':  # xmin, ymin, xmax, ymax
        # top left
        tl_intersect = torch.max(bboxes_a[:, np.newaxis, :2], bboxes_b[:, :2]) # of shape `(N,K,2)`
        # bottom right
        br_intersect = torch.min(bboxes_a[:, np.newaxis, 2:], bboxes_b[:, 2:])
        bb_a = bboxes_a[:, 2:] - bboxes_a[:, :2]
        bb_b = bboxes_b[:, 2:] - bboxes_b[:, :2]
    elif fmt.lower() == 'coco':  # xmin, ymin, w, h
        tl_intersect = torch.max((bboxes_a[:, np.newaxis, :2] - bboxes_a[:, np.newaxis, 2:] / 2),
                       (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2))
        # bottom right
        br_intersect = torch.min((bboxes_a[:, np.newaxis, :2] + bboxes_a[:, np.newaxis, 2:] / 2),
                       (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2))
        bb_a = bboxes_a[:, 2:]
        bb_b = bboxes_b[:, 2:]

    area_a = torch.prod(bb_a, 1)
    area_b = torch.prod(bb_b, 1)

    print(f"area_a = {area_a}")
    print(f"area_b = {area_b}")

    if N==K==1:
        ba, bb = bboxes_a[0], bboxes_b[0]
    
    # torch.prod(input, dim, keepdim=False, dtype=None) → Tensor
    # Returns the product of each row of the input tensor in the given dimension dim
    # if tl, br does not form a nondegenerate squre, then the corr. element in the `prod` would be 0
    en = (tl_intersect < br_intersect).type(tl_intersect.type()).prod(dim=2)  # shape `(N,K,2)` ---> shape `(N,K)`

    area_intersect = torch.prod(br_intersect - tl_intersect, 2) * en  # * ((tl < br).all())
    area_union = (area_a[:, np.newaxis] + area_b - area_intersect)

    iou = torch.true_divide(area_intersect, area_union)

    if iou_type.lower() == 'iou':
        return iou

    if fmt.lower() == 'voc':  # xmin, ymin, xmax, ymax
        # top left
        tl_union = torch.min(bboxes_a[:, np.newaxis, :2], bboxes_b[:, :2]) # of shape `(N,K,2)`
        # bottom right
        br_union = torch.max(bboxes_a[:, np.newaxis, 2:], bboxes_b[:, 2:])
    elif fmt.lower() == 'coco':  # xmin, ymin, w, h
        tl_union = torch.min((bboxes_a[:, np.newaxis, :2] - bboxes_a[:, np.newaxis, 2:] / 2),
                       (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2))
        # bottom right
        br_union = torch.max((bboxes_a[:, np.newaxis, :2] + bboxes_a[:, np.newaxis, 2:] / 2),
                       (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2))
    
    # c for covering, of shape `(N,K,2)`
    # the last dim is box width, box hight
    bboxes_c = br_union - tl_union

    area_covering = torch.prod(bboxes_c, 2)  # shape `(N,K)`

    giou = iou - (area_covering - area_union) / area_covering

    if iou_type.lower() == 'giou':
        return giou

    if fmt.lower() == 'voc':  # xmin, ymin, xmax, ymax
        centre_a = (bboxes_a[..., 2 :] + bboxes_a[..., : 2]) / 2
        centre_b = (bboxes_b[..., 2 :] + bboxes_b[..., : 2]) / 2
    elif fmt.lower() == 'coco':  # xmin, ymin, w, h
        centre_a = (bboxes_a[..., : 2] + bboxes_a[..., 2 :]) / 2
        centre_b = (bboxes_b[..., : 2] + bboxes_b[..., 2 :]) / 2

    centre_dist = torch.norm(centre_a[:, np.newaxis] - centre_b, p='fro', dim=2)
    diag_len = torch.norm(bboxes_c, p='fro', dim=2)

    diou = iou - centre_dist.pow(2) / diag_len.pow(2)

    if iou_type.lower() == 'diou':
        return diou

    # bb_a of shape `(N,2)`, bb_b of shape `(K,2)`
    v = torch.einsum('nm,km->nk', bb_a, bb_b)
    v = torch.true_divide(v, (torch.norm(bb_a, p='fro', dim=1)[:,np.newaxis] * torch.norm(bb_b, p='fro', dim=1)))
    v = torch.true_divide(2*torch.acos(v), np.pi).pow(2)
    alpha = (iou>=0.5).type(iou.type())

    ciou = diou - alpha * v

    if iou_type.lower() == 'ciou':
        return ciou


def bboxes_giou(bboxes_a, bboxes_b, fmt='voc'):
    return bboxes_iou(bboxes_a, bboxes_b, fmt, 'giou')


def bboxes_diou(bboxes_a, bboxes_b, fmt='voc'):
    return bboxes_iou(bboxes_a, bboxes_b, fmt, 'diou')


def bboxes_ciou(bboxes_a, bboxes_b, fmt='voc'):
    return bboxes_iou(bboxes_a, bboxes_b, fmt, 'ciou')

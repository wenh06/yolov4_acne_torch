# -*- coding: utf-8 -*-
'''

'''
import torch
import os, sys
from torch.nn import functional as F
from easydict import EasyDict as ED

import numpy as np


def bboxes_iou_test(bboxes_a, bboxes_b, fmt='voc', iou_type='iou'):
    """
    test function for the bboxes_iou function in `train_acne.py`,
    with message printing and plot
    """
    if 'plt' not in dir():
        import matplotlib.pyplot as plt
    if 'cv2' not in dir():
        import cv2
    
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

    # torch.prod(input, dim, keepdim=False, dtype=None) → Tensor
    # Returns the product of each row of the input tensor in the given dimension dim
    # if tl, br does not form a nondegenerate squre, then the corr. element in the `prod` would be 0
    en = (tl_intersect < br_intersect).type(tl_intersect.type()).prod(dim=2)  # shape `(N,K,2)` ---> shape `(N,K)`

    area_intersect = torch.prod(br_intersect - tl_intersect, 2) * en  # * ((tl < br).all())
    area_union = (area_a[:, np.newaxis] + area_b - area_intersect)

    iou = torch.true_divide(area_intersect, area_union)

    # if iou_type.lower() == 'iou':
    #     return iou

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

    print(f"tl_union.shape = {tl_union.shape}")
    print(f"br_union.shape = {br_union.shape}")
    print(f"bboxes_c.shape = {bboxes_c.shape}")

    # if iou_type.lower() == 'giou':
    #     return giou

    if fmt.lower() == 'voc':  # xmin, ymin, xmax, ymax
        centre_a = (bboxes_a[..., 2 :] + bboxes_a[..., : 2]) / 2
        centre_b = (bboxes_b[..., 2 :] + bboxes_b[..., : 2]) / 2
    elif fmt.lower() == 'coco':  # xmin, ymin, w, h
        centre_a = (bboxes_a[..., : 2] + bboxes_a[..., 2 :]) / 2
        centre_b = (bboxes_b[..., : 2] + bboxes_b[..., 2 :]) / 2

    centre_dist = torch.norm(centre_a[:, np.newaxis] - centre_b, p='fro', dim=2)
    diag_len = torch.norm(bboxes_c, p='fro', dim=2)

    diou = iou - centre_dist.pow(2) / diag_len.pow(2)

    # if iou_type.lower() == 'diou':
    #     return diou

    # bb_a of shape `(N,2)`, bb_b of shape `(K,2)`
    v = torch.einsum('nm,km->nk', bb_a, bb_b)
    v = torch.true_divide(v, (torch.norm(bb_a, p='fro', dim=1)[:,np.newaxis] * torch.norm(bb_b, p='fro', dim=1)))
    v = torch.true_divide(2*torch.acos(v), np.pi).pow(2)
    alpha = (torch.true_divide(v, 1-iou+v))*((iou>=0.5).type(iou.type()))

    ciou = diou - alpha * v

    if N==K==1:
        print("\n"+"*"*50)
        print(f"bboxes_a = {bboxes_a}")
        print(f"bboxes_b = {bboxes_b}")

        print(f"area_a = {area_a}")
        print(f"area_b = {area_b}")

        print(f"area_intersect = {area_intersect}")
        print(f"area_union = {area_union}")

        print(f"tl_intersect = {tl_intersect}")
        print(f"br_intersect = {br_intersect}")
        print(f"tl_union = {tl_union}")
        print(f"br_union = {br_union}")

        print(f"area_covering (area of bboxes_c) = {area_covering}")
        
        print(f"centre_dist = {centre_dist}")
        print(f"diag_len = {diag_len}")

        print("for computing ciou")
        print(f"v = {v}")
        print(f"alpha = {alpha}")

        bc = ED({"xmin":tl_union.numpy().astype(int)[0][0][0], "ymin":tl_union.numpy().astype(int)[0][0][1], "xmax":br_union.numpy().astype(int)[0][0][0], "ymax":br_union.numpy().astype(int)[0][0][1]})
        adjust_x = bc.xmin - int(0.25*(bc.xmax-bc.xmin))
        adjust_y = bc.ymin - int(0.25*(bc.ymax-bc.ymin))

        print(f"adjust_x = {adjust_x}")
        print(f"adjust_y = {adjust_y}")

        bc.xmin, bc.ymin, bc.xmax, bc.ymax = bc.xmin-adjust_x, bc.ymin-adjust_y, bc.xmax-adjust_x, bc.ymax-adjust_y
        
        ba, bb = bboxes_a.numpy().astype(int)[0], bboxes_b.numpy().astype(int)[0]
        if fmt.lower() == 'voc':  # xmin, ymin, xmax, ymax
            ba = ED({"xmin":ba[0]-adjust_x, "ymin":ba[1]-adjust_y, "xmax":ba[2]-adjust_x, "ymax":ba[3]-adjust_y})
            bb = ED({"xmin":bb[0]-adjust_x, "ymin":bb[1]-adjust_y, "xmax":bb[2]-adjust_x, "ymax":bb[3]-adjust_y})
        elif fmt.lower() == 'coco':  # xmin, ymin, w, h
            ba = ED({"xmin":ba[0]-adjust_x, "ymin":ba[1], "xmax":ba[0]+ba[2]-adjust_x, "ymax":ba[1]+ba[3]-adjust_y})
            bb = ED({"xmin":bb[0]-adjust_x, "ymin":bb[1], "xmax":bb[0]+bb[2]-adjust_x, "ymax":bb[1]+bb[3]-adjust_y})

        print(f"ba = {ba}")
        print(f"bb = {bb}")
        print(f"bc = {bc}")

        plane = np.full(shape=(int(1.5*(bc.ymax-bc.ymin)),int(1.5*(bc.xmax-bc.xmin)),3), fill_value=255, dtype=np.uint8)
        img_with_boxes = plane.copy()

        line_size = 1
        cv2.rectangle(img_with_boxes, (ba.xmin, ba.ymin), (ba.xmax, ba.ymax), (0, 255, 0), line_size)
        cv2.rectangle(img_with_boxes, (bb.xmin, bb.ymin), (bb.xmax, bb.ymax), (0, 0, 255), line_size)
        cv2.rectangle(img_with_boxes, (bc.xmin, bc.ymin), (bc.xmax, bc.ymax), (255, 0, 0), line_size)

        plt.figure(figsize=(7,7))
        plt.imshow(img_with_boxes)
        plt.show()

        print(f"iou = {iou}")
        print(f"giou = {giou}")
        print(f"diou = {diou}")
        print(f"ciou = {ciou}")

    if iou_type.lower() == 'ciou':
        return ciou
    elif iou_type.lower() == 'diou':
        return diou
    elif iou_type.lower() == 'giou':
        return giou
    elif iou_type.lower() == 'iou':
        return iou


def original_iou_test(bboxes_a, bboxes_b, xyxy=True):
    """
    test function for the original iou function in `train.py`
    """
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    if isinstance(bboxes_a, np.ndarray):
        bboxes_a = torch.Tensor(bboxes_a)
    if isinstance(bboxes_b, np.ndarray):
        bboxes_b = torch.Tensor(bboxes_a)
    
    N, K = bboxes_a.shape[0], bboxes_b.shape[0]
    # if N, K all equal 1, then plot
    
    # top left
    if xyxy:
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        # bottom right
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        tl = torch.max((bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
                       (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2))
        # bottom right
        br = torch.min((bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
                       (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2))

        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)
    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())

    print(f"tl.shape = {tl.shape}")
    print(f"br.shape = {br.shape}")
    print(f"area_a.shape = {area_a.shape}")
    print(f"area_b.shape = {area_b.shape}")
    print(f"en.shape = {en.shape}")
    print(f"area_i.shape = {area_i.shape}")

    if N == K == 1:
        pass

    return area_i / (area_a[:, None] + area_b - area_i)
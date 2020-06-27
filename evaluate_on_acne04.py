
"""
"""
import os
import time
import glob
from random import shuffle
from numbers import Real
import matplotlib.pyplot as plt

import torch
import cv2
from easydict import EasyDict as ED

from cfg_acne04 import Cfg
from tool.utils import nms_cpu
from tool.torch_utils import do_detect


_CV2_GREEN = (0, 255, 0)


def detect_and_draw(model, image_path, conf_thresh=0.5, show=False, **kwargs):
    """
    """
    model.eval()
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image, (Cfg.width, Cfg.height))

    detected_boxes = do_detect(model, resized, conf_thresh, 0.5, False)[0]
    detected_acne_num = len(detected_boxes)
    img_with_boxes, detected_boxes, detected_scores = plot_boxes_cv2(image, detected_boxes)
    detected_severity = to_severity(detected_acne_num)

    if show:
        title = kwargs.get("title", None)
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=kwargs.get("figsize", (30, 18)))
        ax1.imshow(img_with_boxes)
        ax2.imshow(image)
        if title:
            ax1.set_title(title+f'\ndetected number {detected_acne_num}', y=-0.1, fontsize=20)
            ax1.axis("off")
            ax2.set_title('original image', y=-0.1, fontsize=20)
            ax2.axis("off")
        if kwargs.get("savefig", False):
            fmt = kwargs.get("fmt", "png")
            plt.savefig(f"./fig/{title or str(int(time.time()))}.{fmt}", bbox_inches='tight', transparent=True)
        plt.show()

    return detected_acne_num, detected_boxes, detected_scores, detected_severity


def plot_boxes_cv2(img, boxes, color=_CV2_GREEN):
    """
    """
    img_with_boxes = np.copy(img)
    boxes_on_img = []
    box_scores = []
    
    line_size = max(1, int(max(img_with_boxes.shape[:2]) / 500))
    font_size = line_size + 1

    width = img_with_boxes.shape[1]
    height = img_with_boxes.shape[0]
    for i in range(len(boxes)):
        box = boxes[i]
        x1 = int((box[0] - box[2] / 2.0) * width)
        y1 = int((box[1] - box[3] / 2.0) * height)
        x2 = int((box[0] + box[2] / 2.0) * width)
        y2 = int((box[1] + box[3] / 2.0) * height)
        boxes_on_img.append([x1,y1,x2,y2])
        box_scores.append(box[5])

        if color:
            rgb = color
        else:
            rgb = _CV2_GREEN
        
        # img_with_boxes = cv2.putText(img_with_boxes, 'acne', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.2, rgb, font_size)
        img_with_boxes = cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), rgb, line_size)

    return img_with_boxes, boxes_on_img, box_scores


def to_severity(lesions_num:Real) -> int:
    """
    The Hayashi criterion
    The appropriate divisions of inflammatory eruptions of half of the face to decide classifications were:     0-5, "mild";
        6-20, "moderate";
        21-50, "severe";
        >=50, "very severe"
    Reference:
    [1] Hayashi N, Akamatsu H, Kawashima M, et al. Establishment of grading criteria for acne severity[J]. The Journal of dermatology, 2008, 35(5): 255-260.
    [2] https://www.ncbi.nlm.nih.gov/pubmed/18477223
    """
    if lesions_num <= 5:
        return 0
    elif lesions_num <= 20:
        return 1
    elif lesions_num <= 50:
        return 2
    else:
        return 3


def inference(model, image_path, conf_thresh=[0.5], use_cuda=False):
    """
    inference on single image
    """
    model.eval()
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image, (Cfg.width, Cfg.height))
    resized = torch.from_numpy(resized.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)

    if use_cuda:
        resized = resized.cuda()
    resized = torch.autograd.Variable(resized)

    raw_output = model(resized)
    raw_output = raw_output.cpu().detach().numpy()

    box_array = raw_output[:, :, :4]
    confs = raw_output[:, :, 4:]

    # [batch, num, num_classes] --> [batch, num]
    max_conf = np.max(confs, axis=2)
    max_id = np.argmax(confs, axis=2)

    detected_boxes = {}
    for k in conf_thresh:
        detected_boxes[k] = []
        argwhere = max_conf[0] > k
        l_box_array = box_array[0, argwhere, :]
        l_max_conf = max_conf[0, argwhere]
        l_max_id = max_id[0, argwhere]

        keep = nms_cpu(l_box_array, l_max_conf, 0.5)
        
        if (keep.size > 0):
            l_box_array = l_box_array[keep, :]
            l_max_conf = l_max_conf[keep]
            l_max_id = l_max_id[keep]

            for j in range(l_box_array.shape[0]):
                detected_boxes[k].append([l_box_array[j, 0], l_box_array[j, 1], l_box_array[j, 2], l_box_array[j, 3], l_max_conf[j], l_max_conf[j], l_max_id[j]])

    detected_acne_num = {k:len(v) for k,v in detected_boxes.items()}
    detected_severity = {k:to_severity(v) for k,v in detected_acne_num.items()}

    ret = {
        k: {"acne_num": detected_acne_num[k], "severity": detected_severity[k]} \
            for k in conf_thresh
    }

    return ret

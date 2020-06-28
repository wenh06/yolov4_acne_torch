# -*- coding: utf-8 -*-
'''
train acne detector using the enhanced ACNE04 dataset

More reference:
[1] https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
'''
import time
import logging
import os, sys
from collections import deque

import cv2
from tqdm import tqdm
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from torch.nn import functional as F
from tensorboardX import SummaryWriter
from easydict import EasyDict as ED

from dataset_acne04 import ACNE04
from cfg_acne04 import Cfg
from models import Yolov4
from tool.utils_iou import (
    bboxes_iou, bboxes_giou, bboxes_diou, bboxes_ciou,
)
from tool.utils import post_processing, plot_boxes_cv2
from tool.tv_reference.utils import MetricLogger
from tool.tv_reference.utils import collate_fn as val_collate
from tool.tv_reference.coco_utils import get_coco_api_from_dataset
from tool.tv_reference.coco_eval import CocoEvaluator


DAS = True


class Yolo_loss(nn.Module):
    def __init__(self, n_classes=1, n_anchors=3, device=None, batch=2, iou_type='iou'):
        super(Yolo_loss, self).__init__()
        self.device = device
        self.strides = [8, 16, 32]
        image_size = 608
        self.n_classes = n_classes
        self.n_anchors = n_anchors
        self.iou_type = iou_type

        self.anchors = [[12, 16], [19, 36], [40, 28], [36, 75], [76, 55], [72, 146], [142, 110], [192, 243], [459, 401]]
        self.anch_masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        self.ignore_thre = 0.5

        self.masked_anchors, self.ref_anchors, self.grid_x, self.grid_y, self.anchor_w, self.anchor_h = [], [], [], [], [], []

        for i in range(3):
            all_anchors_grid = [(w / self.strides[i], h / self.strides[i]) for w, h in self.anchors]
            masked_anchors = np.array([all_anchors_grid[j] for j in self.anch_masks[i]], dtype=np.float32)
            ref_anchors = np.zeros((len(all_anchors_grid), 4), dtype=np.float32)
            ref_anchors[:, 2:] = np.array(all_anchors_grid, dtype=np.float32)
            ref_anchors = torch.from_numpy(ref_anchors)
            # calculate pred - xywh obj cls
            fsize = image_size // self.strides[i]
            grid_x = torch.arange(fsize, dtype=torch.float).repeat(batch, 3, fsize, 1).to(device)
            grid_y = torch.arange(fsize, dtype=torch.float).repeat(batch, 3, fsize, 1).permute(0, 1, 3, 2).to(device)
            anchor_w = torch.from_numpy(masked_anchors[:, 0]).repeat(batch, fsize, fsize, 1).permute(0, 3, 1, 2).to(
                device)
            anchor_h = torch.from_numpy(masked_anchors[:, 1]).repeat(batch, fsize, fsize, 1).permute(0, 3, 1, 2).to(
                device)

            self.masked_anchors.append(masked_anchors)
            self.ref_anchors.append(ref_anchors)
            self.grid_x.append(grid_x)
            self.grid_y.append(grid_y)
            self.anchor_w.append(anchor_w)
            self.anchor_h.append(anchor_h)

    def build_target(self, pred, labels, batchsize, fsize, n_ch, output_id):
        # target assignment
        tgt_mask = torch.zeros(batchsize, self.n_anchors, fsize, fsize, 4 + self.n_classes).to(device=self.device)
        obj_mask = torch.ones(batchsize, self.n_anchors, fsize, fsize).to(device=self.device)
        tgt_scale = torch.zeros(batchsize, self.n_anchors, fsize, fsize, 2).to(self.device)
        target = torch.zeros(batchsize, self.n_anchors, fsize, fsize, n_ch).to(self.device)

        # labels = labels.cpu().data
        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects

        truth_x_all = (labels[:, :, 2] + labels[:, :, 0]) / (self.strides[output_id] * 2)
        truth_y_all = (labels[:, :, 3] + labels[:, :, 1]) / (self.strides[output_id] * 2)
        truth_w_all = (labels[:, :, 2] - labels[:, :, 0]) / self.strides[output_id]
        truth_h_all = (labels[:, :, 3] - labels[:, :, 1]) / self.strides[output_id]
        truth_i_all = truth_x_all.to(torch.int16).cpu().numpy()
        truth_j_all = truth_y_all.to(torch.int16).cpu().numpy()

        for b in range(batchsize):
            n = int(nlabel[b])
            if n == 0:
                continue
            truth_box = torch.zeros(n, 4).to(self.device)
            truth_box[:n, 2] = truth_w_all[b, :n]
            truth_box[:n, 3] = truth_h_all[b, :n]
            truth_i = truth_i_all[b, :n]
            truth_j = truth_j_all[b, :n]

            # calculate iou between truth and reference anchors
            # anchor_ious_all = bboxes_iou(truth_box.cpu(), self.ref_anchors[output_id])
            anchor_ious_all = bboxes_iou(
                truth_box.cpu(),
                self.ref_anchors[output_id],
                fmt='voc',
                # iou_type='iou',
                iou_type=self.iou_type,
            )
            best_n_all = anchor_ious_all.argmax(dim=1)
            best_n = best_n_all % 3
            best_n_mask = ((best_n_all == self.anch_masks[output_id][0]) |
                           (best_n_all == self.anch_masks[output_id][1]) |
                           (best_n_all == self.anch_masks[output_id][2]))

            if sum(best_n_mask) == 0:
                continue

            truth_box[:n, 0] = truth_x_all[b, :n]
            truth_box[:n, 1] = truth_y_all[b, :n]

            # pred_ious = bboxes_iou(pred[b].view(-1, 4), truth_box, xyxy=False)
            pred_ious = bboxes_iou(
                pred[b].view(-1, 4),
                truth_box,
                fmt='yolo',
                # iou_type='iou',
                iou_type=self.iou_type,
            )
            pred_best_iou, _ = pred_ious.max(dim=1)
            pred_best_iou = (pred_best_iou > self.ignore_thre)
            pred_best_iou = pred_best_iou.view(pred[b].shape[:3])
            # set mask to zero (ignore) if pred matches truth
            obj_mask[b] = ~ pred_best_iou

            for ti in range(best_n.shape[0]):
                if best_n_mask[ti] == 1:
                    i, j = truth_i[ti], truth_j[ti]
                    a = best_n[ti]
                    obj_mask[b, a, j, i] = 1
                    tgt_mask[b, a, j, i, :] = 1
                    target[b, a, j, i, 0] = truth_x_all[b, ti] - truth_x_all[b, ti].to(torch.int16).to(torch.float)
                    target[b, a, j, i, 1] = truth_y_all[b, ti] - truth_y_all[b, ti].to(torch.int16).to(torch.float)
                    target[b, a, j, i, 2] = torch.log(
                        truth_w_all[b, ti] / torch.Tensor(self.masked_anchors[output_id])[best_n[ti], 0] + 1e-16)
                    target[b, a, j, i, 3] = torch.log(
                        truth_h_all[b, ti] / torch.Tensor(self.masked_anchors[output_id])[best_n[ti], 1] + 1e-16)
                    target[b, a, j, i, 4] = 1
                    target[b, a, j, i, 5 + labels[b, ti, 4].to(torch.int16).cpu().numpy()] = 1
                    tgt_scale[b, a, j, i, :] = torch.sqrt(2 - truth_w_all[b, ti] * truth_h_all[b, ti] / fsize / fsize)
        return obj_mask, tgt_mask, tgt_scale, target

    def forward(self, xin, labels=None):
        loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2 = 0, 0, 0, 0, 0, 0
        for output_id, output in enumerate(xin):
            batchsize = output.shape[0]
            fsize = output.shape[2]
            n_ch = 5 + self.n_classes

            output = output.view(batchsize, self.n_anchors, n_ch, fsize, fsize)
            output = output.permute(0, 1, 3, 4, 2)  # .contiguous()

            # logistic activation for xy, obj, cls
            output[..., np.r_[:2, 4:n_ch]] = torch.sigmoid(output[..., np.r_[:2, 4:n_ch]])

            pred = output[..., :4].clone()
            pred[..., 0] += self.grid_x[output_id]
            pred[..., 1] += self.grid_y[output_id]
            pred[..., 2] = torch.exp(pred[..., 2]) * self.anchor_w[output_id]
            pred[..., 3] = torch.exp(pred[..., 3]) * self.anchor_h[output_id]

            obj_mask, tgt_mask, tgt_scale, target = self.build_target(
                pred, labels, batchsize, fsize, n_ch, output_id
            )

            # loss calculation
            output[..., 4] *= obj_mask
            output[..., np.r_[0:4, 5:n_ch]] *= tgt_mask
            output[..., 2:4] *= tgt_scale

            target[..., 4] *= obj_mask
            target[..., np.r_[0:4, 5:n_ch]] *= tgt_mask
            target[..., 2:4] *= tgt_scale

            loss_xy += F.binary_cross_entropy(
                input=output[..., :2],
                target=target[..., :2],
                weight=tgt_scale*tgt_scale,
                size_average=False,
            )
            loss_wh += F.mse_loss(input=output[..., 2:4], target=target[..., 2:4], size_average=False) / 2
            loss_obj += F.binary_cross_entropy(input=output[..., 4], target=target[..., 4], size_average=False)
            loss_cls += F.binary_cross_entropy(input=output[..., 5:], target=target[..., 5:], size_average=False)
            loss_l2 += F.mse_loss(input=output, target=target, size_average=False)

        loss = loss_xy + loss_wh + loss_obj + loss_cls

        return loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2


def collate(batch):
    images = []
    bboxes = []
    for img, box in batch:
        images.append([img])
        bboxes.append([box])
    images = np.concatenate(images, axis=0)
    images = images.transpose(0, 3, 1, 2)
    images = torch.from_numpy(images).div(255.0)
    bboxes = np.concatenate(bboxes, axis=0)
    bboxes = torch.from_numpy(bboxes)
    return images, bboxes


def train(model, device, config, epochs=5, batch_size=1, save_ckpt=True, log_step=20, logger=None, img_scale=0.5):
    """
    """
    train_dataset = ACNE04(label_path=config.train_label, cfg=config, train=True)
    val_dataset = ACNE04(label_path=config.val_label, cfg=config, train=False)

    n_train = len(train_dataset)
    n_val = len(val_dataset)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch // config.subdivisions,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True,  # setting False would result in error
        collate_fn=collate,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.batch // config.subdivisions,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True,  # setting False would result in error
        collate_fn=val_collate,
    )

    writer = SummaryWriter(
        log_dir=config.TRAIN_TENSORBOARD_DIR,
        filename_suffix=f'OPT_{config.TRAIN_OPTIMIZER}_LR_{config.learning_rate}_BS_{config.batch}_Sub_{config.subdivisions}_Size_{config.width}',
        comment=f'OPT_{config.TRAIN_OPTIMIZER}_LR_{config.learning_rate}_BS_{config.batch}_Sub_{config.subdivisions}_Size_{config.width}',
    )
    
    max_itr = config.TRAIN_EPOCHS * n_train
    # global_step = cfg.TRAIN_MINEPOCH * n_train
    global_step = 0
    if logger:
        logger.info(f'''Starting training:
            Epochs:          {epochs}
            Batch size:      {config.batch}
            Subdivisions:    {config.subdivisions}
            Learning rate:   {config.learning_rate}
            Training size:   {n_train}
            Validation size: {n_val}
            Checkpoints:     {save_ckpt}
            Device:          {device.type}
            Images size:     {config.width}
            Optimizer:       {config.TRAIN_OPTIMIZER}
            Dataset classes: {config.classes}
            Train label path:{config.train_label}
            Pretrained:      {config.pretrained}
        ''')

    # learning rate setup
    def burnin_schedule(i):
        if i < config.burn_in:
            factor = pow(i / config.burn_in, 4)
        elif i < config.steps[0]:
            factor = 1.0
        elif i < config.steps[1]:
            factor = 0.1
        else:
            factor = 0.01
        return factor

    if config.TRAIN_OPTIMIZER.lower() == 'adam':
        optimizer = optim.Adam(
            params=model.parameters(),
            lr=config.learning_rate / config.batch,
            betas=(0.9, 0.999),
            eps=1e-08,
        )
    elif config.TRAIN_OPTIMIZER.lower() == 'sgd':
        optimizer = optim.SGD(
            params=model.parameters(),
            lr=config.learning_rate / config.batch,
            momentum=config.momentum,
            weight_decay=config.decay,
        )
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, burnin_schedule)

    criterion = Yolo_loss(
        n_classes=config.classes,
        device=device,
        batch=config.batch // config.subdivisions,
        iou_type=config.iou_type,
    )
    # scheduler = ReduceLROnPlateau(optimizer, mode='max', verbose=True, patience=6, min_lr=1e-7)
    # scheduler = CosineAnnealingWarmRestarts(optimizer, 0.001, 1e-6, 20)

    save_prefix = 'Yolov4_epoch'
    saved_models = deque()
    model.train()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_step = 0

        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img', ncols=100) as pbar:
            for i, batch in enumerate(train_loader):
                global_step += 1
                epoch_step += 1
                images = batch[0]
                bboxes = batch[1]

                images = images.to(device=device, dtype=torch.float32)
                bboxes = bboxes.to(device=device)

                bboxes_pred = model(images)
                loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2 = criterion(bboxes_pred, bboxes)
                # loss = loss / config.subdivisions
                loss.backward()

                epoch_loss += loss.item()

                if global_step  % config.subdivisions == 0:
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()

                if global_step % (log_step * config.subdivisions) == 0:
                    writer.add_scalar('train/Loss', loss.item(), global_step)
                    writer.add_scalar('train/loss_xy', loss_xy.item(), global_step)
                    writer.add_scalar('train/loss_wh', loss_wh.item(), global_step)
                    writer.add_scalar('train/loss_obj', loss_obj.item(), global_step)
                    writer.add_scalar('train/loss_cls', loss_cls.item(), global_step)
                    writer.add_scalar('train/loss_l2', loss_l2.item(), global_step)
                    writer.add_scalar('lr', scheduler.get_lr()[0] * config.batch, global_step)
                    pbar.set_postfix(**{
                        'loss (batch)': loss.item(),
                        'loss_xy': loss_xy.item(),
                        'loss_wh': loss_wh.item(),
                        'loss_obj': loss_obj.item(),
                        'loss_cls': loss_cls.item(),
                        'loss_l2': loss_l2.item(),
                        'lr': scheduler.get_lr()[0] * config.batch
                    })
                    if logger:
                        logger.info(f'Train step_{global_step}: loss : {loss.item()},loss xy : {loss_xy.item()}, loss wh : {loss_wh.item()}, loss obj : {loss_obj.item()}, loss cls : {loss_cls.item()}, loss l2 : {loss_l2.item()}, lr : {scheduler.get_lr()[0] * config.batch}')

                # TODO: eval for each epoch using `evaluate`
                # evaluate(model, val_loader, device, logger)
                # model.train()

                pbar.update(images.shape[0])

            if save_ckpt:
                try:
                    os.mkdir(config.checkpoints)
                    if logger:
                        logger.info('Created checkpoint directory')
                except OSError:
                    pass
                save_path = os.path.join(config.checkpoints, f'{save_prefix}{epoch + 1}.pth')
                torch.save(model.state_dict(), save_path)
                saved_models.append(save_path)
                # remove outdated models
                if len(saved_models) > config.keep_checkpoint_max > 0:
                    model_to_remove = saved_models.popleft()
                    try:
                        os.remove(model_to_remove)
                    except:
                        logger.info(f'failed to remove {model_to_remove}')
                if logger:
                    logger.info(f'Checkpoint {epoch + 1} saved!')

    writer.close()


@torch.no_grad()
def evaluate(model, data_loader, cfg, device, logger, **kwargs):
    """ finished, has bugs
    """
    cpu_device = torch.device("cpu")
    model.eval()
    header = 'Test:'

    coco = get_coco_api_from_dataset(data_loader.dataset)
    coco_evaluator = CocoEvaluator(coco, iou_types = ["bbox"], bbox_fmt='coco')

    for images, targets in data_loader:
        model_input = [[cv2.resize(img, (cfg.w, cfg.h))] for img in images]
        model_input = np.concatenate(model_input, axis=0)
        model_input = model_input.transpose(0, 3, 1, 2)
        model_input = torch.from_numpy(model_input).div(255.0)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(model_input)

        # outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        outputs = outputs.cpu().detach().numpy()
        res = {}
        for img, target, output in zip(images, targets, outputs):
            img_height, img_width = img.shape[:2]
            boxes = output[...,:4].copy()  # output boxes in yolo format
            boxes[...,:2] = boxes[...,:2] - boxes[...,2:]/2  # to coco format
            boxes[...,0] = boxes[...,0]*img_width
            boxes[...,1] = boxes[...,1]*img_height
            boxes[...,2] = boxes[...,2]*img_width
            boxes[...,3] = boxes[...,3]*img_height
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(np.zeros((len(output),)), dtype=torch.int64)
            scores = torch.as_tensor(output[...,-1], dtype=torch.float32)
            res[target["image_id"].item()] = {
                "boxes": boxes,
                "scores": scores,
                "labels": labels,
            }

        debug = kwargs.get("debug", [])
        if isinstance(debug, str):
            debug = [debug]
        debug = [item.lower() for item in debug]
        if 'iou' in debug:
            from tool.utils_iou_test import bboxes_iou_test
            ouput_boxes = np.array(post_processing(None, 0.5, 0.5, outputs)[0])[...,:4]
            img_height, img_width = images[0].shape[:2]
            ouput_boxes[...,0] = ouput_boxes[...,0] * img_width
            ouput_boxes[...,1] = ouput_boxes[...,1] * img_height
            ouput_boxes[...,2] = ouput_boxes[...,2] * img_width
            ouput_boxes[...,3] = ouput_boxes[...,3] * img_height
            # coco format to yolo format
            truth_boxes = targets[0]['boxes'].numpy().copy()
            truth_boxes[...,:2] = truth_boxes[...,:2] + truth_boxes[...,2:]/2
            iou = bboxes_iou_test(torch.Tensor(ouput_boxes), torch.Tensor(truth_boxes), fmt='yolo')
            print(f"iou of first image = {iou}")
        if len(debug) > 0:
            return
        
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time

    # gather the stats from all processes
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    return coco_evaluator


def get_args(**kwargs):
    """
    """
    pretrained_detector = '/mnt/wenhao71/workspace/yolov4_acne_torch/pretrained/yolov4.pth'
    cfg = kwargs
    parser = argparse.ArgumentParser(
        description='Train the Model on images and target masks',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument(
    #     '-b', '--batch-size',
    #     metavar='B', type=int, nargs='?', default=2,
    #     help='Batch size',
    #     dest='batchsize')
    parser.add_argument(
        '-l', '--learning-rate',
        metavar='LR', type=float, nargs='?', default=0.001,
        help='Learning rate',
        dest='learning_rate')
    parser.add_argument(
        '-f', '--load',
        dest='load', type=str, default=pretrained_detector,
        help='Load model from a .pth file')
    parser.add_argument(
        '-g', '--gpu',
        metavar='G', type=str, default='0',
        help='GPU',
        dest='gpu')
    # `dataset_dir` and `pretrained` already set in cfg_acne04.py
    # parser.add_argument(
    #     '-dir', '--data-dir',
    #     type=str, default=None,
    #     help='dataset dir', dest='dataset_dir')
    # parser.add_argument(
    #     '-pretrained',
    #     type=str, default=None,
    #     help='pretrained yolov4.conv.137')
    parser.add_argument(
        '-classes',
        type=int, default=1,
        help='dataset classes')
    # parser.add_argument(
    #     '-train_label_path',
    #     dest='train_label', type=str, default='train.txt',
    #     help="train label path")
    parser.add_argument(
        '-iou-type', type=str, default='iou',
        help='iou type (iou, giou, diou, ciou)',
        dest='iou_type')
    parser.add_argument(
        '-keep-checkpoint-max', type=int, default=10,
        help='maximum number of checkpoints to keep. If set 0, all checkpoint files are kept',
        dest='keep_checkpoint_max')
    parser.add_argument(
        '-optimizer', type='str', default='adam',
        help='training optimizer',
        dest='TRAIN_OPTIMIZER')
    
    args = vars(parser.parse_args())

    cfg.update(args)
    
    return ED(cfg)


def init_logger(log_file=None, log_dir=None, mode='a', verbose=0):
    """
    """
    import datetime
    def get_date_str():
        now = datetime.datetime.now()
        return now.strftime('%Y-%m-%d_%H-%M')

    if log_dir is None:
        log_dir = '~/temp/log/'
    if log_file is None:
        log_file = f'log_{get_date_str()}.txt'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, log_file)
    print(f'log file path: {log_file}')

    logger = logging.getLogger('Yolov4-ACNE04')

    c_handler = logging.StreamHandler(sys.stdout)
    f_handler = logging.FileHandler(log_file)

    if verbose >= 2:
        print("levels of c_handler and f_handler are set DEBUG")
        c_handler.setLevel(logging.DEBUG)
        f_handler.setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    elif verbose >= 1:
        print("level of c_handler is set INFO, level of f_handler is set DEBUG")
        c_handler.setLevel(logging.INFO)
        f_handler.setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    else:
        print("levels of c_handler and f_handler are set WARNING")
        c_handler.setLevel(logging.WARNING)
        f_handler.setLevel(logging.WARNING)
        logger.setLevel(logging.WARNING)

    c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger


def _get_date_str():
    import datetime
    now = datetime.datetime.now()
    return now.strftime('%Y-%m-%d_%H-%M')


"""
torch, torch vision, cu compatibility:
https://download.pytorch.org/whl/torch_stable.html
https://download.pytorch.org/whl/cu100/torch-1.3.1%2Bcu100-cp36-cp36m-linux_x86_64.whl
"""

if __name__ == "__main__":
    cfg = get_args(**Cfg)
    # os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu
    if not DAS:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cuda')
    log_dir = cfg.TRAIN_TENSORBOARD_DIR
    logger = init_logger(log_dir=log_dir)
    logger.info(f"\n{'*'*20}   Start Training   {'*'*20}\n")
    logger.info(f'Using device {device}')
    logger.info(f"Using torch of version {torch.__version__}")
    logger.info(f'with configuration {cfg}')
    print(f"\n{'*'*20}   Start Training   {'*'*20}\n")
    print(f'Using device {device}')
    print(f"Using torch of version {torch.__version__}")
    print(f'with configuration {cfg}')

    model = Yolov4(cfg.pretrained, n_classes=cfg.classes)

    if not DAS and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    if not DAS:
        model.to(device=device)
    else:
        model.cuda()

    try:
        train(
            model=model,
            config=cfg,
            epochs=cfg.TRAIN_EPOCHS,
            device=device,
            logger=logger,
        )
    except KeyboardInterrupt:
        torch.save(model.state_dict(), os.path.join(cfg.checkpoints, 'INTERRUPTED.pth'))
        logger.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

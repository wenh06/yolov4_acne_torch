"""
"""
from easydict import EasyDict as ED

Cfg = ED()
Cfg.batch = 2
Cfg.subdivisions = 1
Cfg.width = 608
Cfg.height = 608
Cfg.channels = 3
Cfg.momentum = 0.949
Cfg.decay = 0.0005
Cfg.angle = 0
Cfg.saturation = 1.5
Cfg.exposure = 1.5
Cfg.hue = .1

Cfg.learning_rate = 0.00261
Cfg.burn_in = 1000
Cfg.max_batches = 500500
Cfg.steps = [600000, 800000]
Cfg.policy = Cfg.steps
Cfg.scales = .1, .1

Cfg.cutmix = 0
Cfg.mosaic = 1

Cfg.letter_box = 0
Cfg.jitter = 0.2
Cfg.classes = 1
Cfg.track = 0
Cfg.w = Cfg.width
Cfg.h = Cfg.height
Cfg.flip = 1
Cfg.blur = 0
Cfg.gaussian = 0
Cfg.boxes = 200  # box num
Cfg.TRAIN_EPOCHS = 300000
Cfg.train_label = '/mnt/wenhao71/data/acne_article_data/train.csv'
Cfg.val_label = '/mnt/wenhao71/data/acne_article_data/val.csv'
Cfg.TRAIN_OPTIMIZER = 'adam'
'''
original data generator accepted format:
image_path1 x1,y1,x2,y2,id x1,y1,x2,y2,id x1,y1,x2,y2,id ...
image_path2 x1,y1,x2,y2,id x1,y1,x2,y2,id x1,y1,x2,y2,id ...
...
'''

if Cfg.mosaic and Cfg.cutmix:
    Cfg.mixup = 4
elif Cfg.cutmix:
    Cfg.mixup = 2
elif Cfg.mosaic:
    Cfg.mixup = 3

# Cfg.checkpoints = 'checkpoints'
Cfg.checkpoints = '/mnt/wenhao71/workspace/yolov4_acne_torch/saved_models/'
Cfg.TRAIN_TENSORBOARD_DIR = '/mnt/wenhao71/workspace/yolov4_acne_torch/log/'

Cfg.dataset_dir = '/mnt/wenhao71/data/acne_article_data/filtered_images/'

# yolov4conv137weight
Cfg.pretrained = '/mnt/wenhao71/workspace/yolov4_acne_torch/pretrained/yolov4.conv.137.pth'

Cfg.iou_type = 'iou'

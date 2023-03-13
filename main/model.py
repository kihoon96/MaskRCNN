import numpy as np
import cv2

import torch
import torch.nn as nn
from torch.nn import functional as F

from nets.fpn import FPN
from nets.rpn import RPN
from utils.anchor_generators import generate_all_anchors, all_anchors_to_image_scale
from utils.IoU import get_IoU, area, get_IoU_tensor

from config import cfg

#from nets.resnet import ResNetBackbone
#from nets.module import PoseNet, Pose2Feat, MeshNet, ParamRegressor
#from nets.loss import CoordLoss, ParamLoss, NormalVectorLoss, EdgeLengthLoss
#import math

import matplotlib.pyplot as plt

class Model(nn.Module):
    def __init__(self, backbone, rpn, head):
        super(Model, self).__init__()
        self.backbone = backbone
        self.rpn = rpn
        self.head = head
        self.anchors = generate_all_anchors() # shape (5, anchor_num(pixels x 3), 4)
        self.anchors = all_anchors_to_image_scale(self.anchors)

        self.trainable_modules = [self.backbone, self.rpn, self.head]
    
    def forward(self, inputs, targets, meta_info, mode):
        gt_bboxes = []
        for bi in range(cfg.train_batch_size):
            padded_gt_bboxes = targets['bboxes'][bi]
            num_valid_bbox = meta_info['num_valid_bbox'][bi]
            gt_bboxes.append(padded_gt_bboxes[:num_valid_bbox])

        fpn_fms = self.backbone(inputs['img'])
        rpn_losses = self.rpn(fpn_fms ,self.anchors, gt_bboxes)

        # anchor visualize
        vis = False
        if vis:
            cvimg = (inputs['img'][0].cpu().numpy().transpose(1,2,0)*255).astype(np.uint8)[:,:,::-1]
            for anchor, scale in zip(self.anchors, (64,32,16,8,4)):
                _anchor = (anchor * scale).cpu().numpy().astype(int)
                _anchor = _anchor.reshape(-1,3,4)

                __anchor = _anchor[len(_anchor)//2+512//scale].reshape(3,-1)
                for ___anchor in __anchor:
                    cvimg = cv2.rectangle(cvimg.copy(), ___anchor[:2], ___anchor[2:], (255,0,0), 3)                
            cv2.imwrite("whole_anchors.png", cvimg)

        losses = {}

        losses['objectiveness_loss'] = rpn_losses[0]
        losses['localization_loss'] = rpn_losses[1]
        losses['total_loss'] = rpn_losses[0] + rpn_losses[1]


        return losses

def init_weights(m):
    if type(m) == nn.ConvTranspose2d:
        nn.init.normal_(m.weight,std=0.001)
    elif type(m) == nn.Conv2d:
        nn.init.normal_(m.weight,std=0.001)
        nn.init.constant_(m.bias, 0)
    elif type(m) == nn.BatchNorm2d:
        nn.init.constant_(m.weight,1)
        nn.init.constant_(m.bias,0)
    elif type(m) == nn.Linear:
        nn.init.normal_(m.weight,std=0.01)
        nn.init.constant_(m.bias,0)

def get_model(mode):
    backbone = FPN()
    rpn = RPN()
    head = None

    # Todo : init net weights
    #if mode == 'train':

    model = Model(backbone, rpn, head)
    return model
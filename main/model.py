import numpy as np
import cv2

import torch
import torch.nn as nn
from torch.nn import functional as F

from nets.fpn import FPN
from utils.anchor_generators import generate_cell_anchors

#from nets.resnet import ResNetBackbone
#from nets.module import PoseNet, Pose2Feat, MeshNet, ParamRegressor
#from nets.loss import CoordLoss, ParamLoss, NormalVectorLoss, EdgeLengthLoss
#import math

class Model(nn.Module):
    def __init__(self, backbone, rpn, head):
        super(Model, self).__init__()
        self.backbone = backbone
        self.rpn = rpn
        self.head = head
        self.cell_anchors = generate_cell_anchors().cuda()

        self.trainable_modules = [self.backbone, self.rpn, self.head]
    
    def forward(self, inputs, targets, meta_info, mode):

        out = self.backbone(inputs['img'])

        anchors = []
        import pdb; pdb.set_trace()
        for fm_name, fm in out.items():
            ## assume rectangle feature map and rectangle input img shape
            fm_res = fm.shape[-1]
            xx = torch.arange(fm_res)
            yy = torch.arange(fm_res)
            grid_y, grid_x = torch.meshgrid(xx,yy)
            coords = torch.stack([grid_x, grid_y]).permute(1,2,0).reshape(-1,2).cuda() # N_pixel x 2

            # 여기서 부터 이해해야함
            anchors.append((self.cell_anchors.reshape(-1,2,2)[None,:,:,:] + coords[:,None,None,:]).reshape(-1,4)) # (N_pixel x N_cell_anchor) x 4
            # p5, p4, p3, p2, p6

        # anchor visualize
        vis = False
        cvimg = (inputs['img'][0].cpu().numpy().transpose(1,2,0)*255).astype(np.uint8)[:,:,::-1]
        if vis:
            for anchor, scale in zip(anchors, (32,16,8,4,64)):
                _anchor = (anchor * scale).cpu().numpy().astype(int)
                _anchor = _anchor.reshape(-1,3,4)

                __anchor = _anchor[len(_anchor)//2+512//scale].reshape(3,-1)
                for ___anchor in __anchor:
                    cvimg = cv2.rectangle(cvimg.copy(), ___anchor[:2], ___anchor[2:], (255,0,0), 3)                
            cv2.imwrite("whole_anchors.png", cvimg)

        import pdb;pdb.set_trace()


        # 





        feat = self.backbone(inputs['img'])
        return

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
    rpn = None
    head = None

    # Todo : init net weights
    #if mode == 'train':

    model = Model(backbone, rpn, head)
    return model
import torch
import torch.nn as nn
from torch.nn import functional as F
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

        self.trainable_modules = [self.backbone, self.rpn, self.head]
    
    def forward(self, inputs, targets, meta_info, mode):
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

def get_model(vertex_num, joint_num, mode):
    backbone = FPN()
    rpn = None
    head = None

    # Todo : init net weights
    #if mode == 'train':

    model = Model(backbone, rpn, head)
    return model
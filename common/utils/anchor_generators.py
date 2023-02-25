import math
import torch
from config import cfg

def generate_cell_anchors():

    # anchor size is 8 for all feature maps

    anchors = []
    size = 8

    area = size**2.0
    for aspect_ratio in cfg.anchor_aspect_ratios:
        w = math.sqrt(area / aspect_ratio)
        h = aspect_ratio * w
        x0, y0, x1, y1 = -w/2.0, -h/2.0, w/2.0, h/2.0
        anchors.append([x0,y0,x1,y1])
        
    return torch.tensor(anchors)
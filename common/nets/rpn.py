import numpy as np
import torch
import torch.nn as nn

class RPN(nn.Module):
    def __init__(self):
        super(RPN, self).__init__()
        in_channels = 1
        out_channels = 256
        cur_channels = out_channels
        num_anchors = 3
        box_dim = 4
        self.inconv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.objectness_logits = nn.Conv2d(cur_channels, num_anchors, kernel_size=1, stride=1)
        self.anchor_deltas = nn.Conv2d(cur_channels, num_anchors * box_dim, kernel_size=1, stride=1)


    def forward(self, features):
        pred_objectness_logits = []
        pred_anchor_deltas = []
        for f in featrues:
            x = self.inconv(f)
            x = self.relu(x)
            pred_objectness_logits.append(self.objetness_logits(x))
            pred_anchor_deltas.append(self.anchor_deltas(x))

        return pred_objectness_logits, pred_anchor_deltas
        

    def init_weights():
        return
    
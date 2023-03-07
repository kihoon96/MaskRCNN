import numpy as np
import torch
import torch.nn as nn

class RPNHead(nn.Module):
    def __init__(self):
        super(RPN, self).__init__()
        in_channels = 256
        out_channels = 256
        cur_channels = out_channels
        num_anchors = 3
        self.box_dim = 4
        self.inconv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.objectness_logits = nn.Conv2d(cur_channels, num_anchors, kernel_size=1, stride=1)
        self.anchor_deltas = nn.Conv2d(cur_channels, num_anchors * box_dim, kernel_size=1, stride=1)


    def forward(self, features):
        pred_objectness_logits = []
        pred_anchor_deltas = []
        for f in features:
            x = self.inconv(f)
            x = self.relu(x)
            pred_objectness_logits.append(self.objectness_logits(x))
            pred_anchor_deltas.append(self.anchor_deltas(x))

        return pred_objectness_logits, pred_anchor_deltas

class RPN(nn.Module):
    def __init__(self):
        self.rpn_head = RPNHead()
        
        self.training = True

    def forward(self, features):
        pred_objectness_logits, pred_anchor_deltas = self.rpn_head(features)
        pred_objectness_logits = [
            # (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A)
            score.permute(0,2,3,1).flatten(1)
            for score in pre_objectness_logits
        ]
        pred_anchor_deltas = [
            # (N, A*B, Hi, Wi) -> (N, A, B, Hi, Wi) -> (N, Hi, Wi, A, B) -> (N, Hi*Wi*A, B)
            x.view(x.shape[0], -1, self.rpn_head.box_dim, x.shape[-2], x.shape[-1])
            .permute(0, 3, 4, 1, 2)
            .flatten(1, -2)
            for x in pred_anchor_deltas
        ]

        if self.training:
            gt_labels, gt_bboxes = 

    def losses(self, anchors, pred_objectness_logits, gt_labels, pred_anchor_deltas, gt_bboxes):
        
        return
        
    def predict_proposals():
        return
    


    def init_weights():
        return
    
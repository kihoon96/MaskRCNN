import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



from utils.smooth_l1_loss import smooth_l1_loss
from utils.anchor_generators import label_anchors, subsample_labels
from utils.delta_transform import get_deltas, apply_deltas

class RPNHead(nn.Module):
    def __init__(self):
        super(RPNHead, self).__init__()

        in_channels = 256
        out_channels = 256
        cur_channels = out_channels
        num_anchors = 3
        self.box_dim = 4
        self.inconv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.objectness_logits = nn.Conv2d(cur_channels, num_anchors, kernel_size=1, stride=1)
        self.anchor_deltas = nn.Conv2d(cur_channels, num_anchors * self.box_dim, kernel_size=1, stride=1)


    def forward(self, features):
        pred_objectness_logits = []
        pred_anchor_deltas = []
        for fm_name, fm in sorted(features.items(), reverse=True):
            x = self.inconv(fm)
            x = self.relu(x)
            pred_objectness_logits.append(self.objectness_logits(x))
            pred_anchor_deltas.append(self.anchor_deltas(x))

        return pred_objectness_logits, pred_anchor_deltas

class RPN(nn.Module):
    def __init__(self):
        super(RPN, self).__init__()

        self.rpn_head = RPNHead()
        
        self.training = True

    def forward(self, features, anchors, gt_bboxes):
        pred_objectness_logits, pred_anchor_deltas = self.rpn_head(features)
        pred_objectness_logits = [
            # (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A)
            score.permute(0,2,3,1).flatten(1)
            for score in pred_objectness_logits
        ]
        for i, logit in enumerate(pred_objectness_logits):
            if i == 0:
                logits = logit
            else:
                logits = torch.cat((logits, logit), dim=1)
        pred_objectness_logits = logits
            
        pred_anchor_deltas = [
            # (N, A*B, Hi, Wi) -> (N, A, B, Hi, Wi) -> (N, Hi, Wi, A, B) -> (N, Hi*Wi*A, B)
            x.view(x.shape[0], -1, self.rpn_head.box_dim, x.shape[-2], x.shape[-1])
            .permute(0, 3, 4, 1, 2)
            .flatten(1, -2)
            for x in pred_anchor_deltas
        ]
        for i, ad in enumerate(pred_anchor_deltas):
            if i == 0:
                ads = ad
            else:
                ads = torch.cat((ads, ad), dim=1)
        pred_anchor_deltas = ads
        
        labels = []
        gt_idx = []
        pos_idx = []
        neg_idx = []
        valid_idx = []
        gt_deltas = []
        object_loss = []
        local_loss = []
        for bi in range(2): #need to change to batch size from cfg
            #device = anchors.cuda.current_device()
            label, gt_i = label_anchors(anchors, gt_bboxes[bi])
            pos_i, neg_i = subsample_labels(label)

            labels.append(label)
            gt_idx.append(gt_i)
            pos_idx.append(pos_i)
            neg_idx.append(neg_i)
            gt_deltas.append(get_deltas(anchors, gt_bboxes[bi][gt_i]))
            valid_idx.append(torch.cat((pos_i, neg_i)))
            object_loss.append(F.binary_cross_entropy_with_logits(pred_objectness_logits[bi][valid_idx[bi]], labels[bi][valid_idx[bi]],reduction="sum"))
            local_loss.append(smooth_l1_loss(pred_anchor_deltas[bi][pos_idx[bi]], gt_deltas[bi][pos_idx[bi]], beta=0.0, reduction="sum"))
        
        #anchor pos&neg sampled visualize
        vis = False
        if vis:
            cvimg = (inputs['img'][bi].cpu().numpy().transpose(1,2,0)*255).astype(np.uint8)[:,:,::-1]
            for gti, gt_bbox in enumerate(gt_bboxes):
                gt_bbox = gt_bbox.cpu().numpy().astype(int)
                cvimg = cv2.rectangle(cvimg.copy(), gt_bbox[:2], gt_bbox[2:], (0,255,0), 3)
                cvimg = cv2.putText(cvimg.copy(), str(gti), gt_bbox[:2] + [5, 20],
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 3)        
            for pos_i in pos_idx:
                pa = self.anchors[pos_i].cpu().numpy().astype(int)
                cvimg = cv2.rectangle(cvimg.copy(), pa[:2], pa[2:], (255,0,0), 3)
            for neg_i in neg_idx:
                na = self.anchors[neg_i].cpu().numpy().astype(int)
                cvimg = cv2.rectangle(cvimg.copy(), na[:2], na[2:], (0,0,255), 3)
        

            cv2.imwrite(f"test.png", cvimg)
            import pdb; pdb.set_trace()

        if self.training:
            gt_labels, gt_bboxes = 1,1
        object_loss = (object_loss[0] + object_loss[1])/2.0
        local_loss = (local_loss[0] + local_loss[1])/2.0
        return object_loss, local_loss

    def losses(self, anchors, pred_objectness_logits, gt_labels, pred_anchor_deltas, gt_bboxes):
        
        return
        
    def predict_proposals():
        return
    


    def init_weights():
        return
    
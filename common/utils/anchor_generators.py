import math
import torch

from config import cfg
from utils.IoU import get_IoU_tensor

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

def generate_all_anchors():
    anchors = []
    cell_anchors = generate_cell_anchors().cuda()
    fm_res_a = cfg.feature_map_resolutions
    for fm_res in fm_res_a:
        ## assume rectangle feature map and rectangle input img shape
        xx = torch.arange(fm_res)
        yy = torch.arange(fm_res)
        grid_y, grid_x = torch.meshgrid(xx,yy)
        coords = torch.stack([grid_x, grid_y]).permute(1,2,0).reshape(-1,2).cuda() # N_pixel x 2

        anchors.append((cell_anchors.reshape(-1,2,2)[None,:,:,:] + coords[:,None,None,:]).reshape(-1,4)) # (N_pixel x N_cell_anchor) x 4

    return anchors

def all_anchors_to_image_scale(anchors):
    for i, stride in enumerate(cfg.anchor_strides):
        new_anchor = (anchors[i] * stride)
        new_anchor = new_anchor.reshape(-1,4)
        if i == 0:
            new_anchors = new_anchor
        else:
            new_anchors = torch.cat((new_anchors, new_anchor), dim=0)

    #clamp bbox out of image size
    torch.clamp(new_anchors, min=0, max=cfg.input_img_shape[0])

    return new_anchors

def label_anchors(anchors, gt_bboxes):
    IoU_neg_thresh = cfg.IoU_neg_thresh
    IoU_pos_thresh = cfg.IoU_pos_thresh
    num_anchors = anchors.shape[0] #N
    num_gt_bboxes = gt_bboxes.shape[0] #M
    labels = torch.full((num_anchors,), -1, dtype=torch.int8)

    ious = get_IoU_tensor(anchors, gt_bboxes) #NxM

    # for each anchor, select gt_bbox with highest iou
    max_ious, gt_idx = ious.max(dim=1)

    # for each gt_bbox, select anchor with highest iou
    gt_anchors_idx = ious.argmax(dim=0)

    # positive anchors for iou >= 0.7 + highest iou for each gt_bbox
    pos_indices = (max_ious >= IoU_pos_thresh).nonzero()
    labels[pos_indices] = 1
    labels[gt_anchors_idx] = 1

    # negative anchors for iou <= 0.3
    neg_indices = (max_ious <= IoU_neg_thresh).nonzero()
    labels[neg_indices] = 0

    return labels, gt_idx

def subsample_labels(labels:torch.Tensor):
    num_samples = cfg.sampling_anchor_num
    positive_fraction = cfg.positive_fraction
    
    positives = (labels == 1).nonzero()
    negatives = (labels == 0).nonzero()

    num_pos = int(num_samples * positive_fraction)
    num_pos = min(positives.numel(), num_pos)
    num_neg = num_samples - num_pos
    num_neg = min(negatives.numel(), num_neg)

    perm1 = torch.randperm(positives.numel())[:num_pos]
    perm2 = torch.randperm(negatives.numel())[:num_neg]

    pos_idx = positives[perm1].squeeze(dim=1)
    neg_idx = negatives[perm2].squeeze(dim=1)

    return pos_idx, neg_idx



    





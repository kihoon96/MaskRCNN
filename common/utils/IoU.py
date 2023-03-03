import numpy
import torch

#xyxy
def area(box):
    return (box[2] - box[0]) * (box[3] - box[1])

#xyxy
def area_tensor(boxes):
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

#xyxy
def get_IoU(box1, box2):
    x1y1 = torch.max(box1[:2], box2[:2])
    x2y2 = torch.min(box1[2:], box2[2:])
    width_height = torch.clamp((x2y2-x1y1), 0)
    inter_area = width_height[0] * width_height[1]
    union_area = area(box1) + area(box2) - inter_area
    IoU = inter_area / union_area
    return IoU
#xyxy
def get_IoU_tensor(boxes1, boxes2):
    x1y1 = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    x2y2 = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    width_height = torch.clamp((x2y2-x1y1), 0)
    inter_area = width_height.prod(dim=2)
    union_area = area_tensor(boxes1)[:, None] + area_tensor(boxes2) - inter_area
    IoU = inter_area / union_area
    return IoU

import torch
from pycocotools.coco import COCO
import json
import numpy as np
import os
import os.path as osp
import cv2

#for visualization
import matplotlib.pyplot as plt
import skimage.io as io

root_path = osp.join('/media', '3_4T', 'COCO')
train_annot_path = osp.join(root_path, 'annotations', 'person_keypoints_train2017.json')
test_annot_path = osp.join(root_path, 'annotations', 'person_keypoints_val2017.json')
train_img_path = osp.join(root_path, 'images', 'train2017')
test_img_path = osp.join(root_path, 'images', 'val2017')
    
class Dataloader_COCO(torch.utils.data.Dataset):
    def __init__(self, mode):
        if mode == 'train':
            self.coco = COCO(train_annot_path)
            self.img_path = train_img_path
        else:
            self.coco = COCO(test_annot_path)
            self.img_path = test_img_path
        self.catIds = self.coco.getCatIds(catNms=['person'])
        self.imgIds = self.coco.getImgIds(catIds=self.catIds)

    def __getitem__(self, index):
        img = self.coco.loadImgs(self.imgIds[index])[0]
        # img bgr to rgb
        img_file_path = osp.join(self.img_path, img['file_name'])
        input_img = cv2.imread(img_file_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)[:,:,::-1]

        annIds = self.coco.getAnnIds(imgIds=img['id'], catIds=self.catIds, iscrowd=False)
        anns = self.coco.loadAnns(annIds)
        bboxes = []
        for ann in anns:
            bboxes.append(ann['bbox'])

        return input_img, bboxes

    def __len__(self):
        return len(self.imgIds)
        
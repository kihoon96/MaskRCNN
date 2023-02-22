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
    
class MSCOCO(torch.utils.data.Dataset):
    def __init__(self, transform, mode):
        
        self.mode = mode
        self.transform = transform

        
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
        # img cv2 bgr to rgb
        img_file_path = osp.join(self.img_path, img['file_name'])
        input_img = np.ascontiguousarray(cv2.imread(img_file_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)[:,:,::-1])
        input_img = self.transform(input_img)

        annIds = self.coco.getAnnIds(imgIds=img['id'], catIds=self.catIds, iscrowd=False)
        anns = self.coco.loadAnns(annIds)


        bboxes = [ann['bbox'] for ann in anns]
        bboxes = np.array(bboxes, dtype = np.float32)

        num_max_bbox = 20
        dummy_bbox = np.zeros((num_max_bbox-bboxes.shape[0], 4))

        num_valid_bbox = float(bboxes.shape[0])
        bboxes = np.concatenate([bboxes, dummy_bbox])

        inputs = {'img': input_img}
        targets = {'bboxes': bboxes}
        meta_info = {'num_valid_bbox': num_valid_bbox}

        if(num_valid_bbox > 20):
            raise ValueError('bbox over 20 need more dummy padding')
        
        return inputs, targets, meta_info

    def __len__(self):
        return len(self.imgIds)
        
import torch
from pycocotools.coco import COCO
import json
import numpy as np
import os
import os.path as osp
import cv2
import copy

from config import cfg
from utils.preprocess import gen_trans

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
        vis = False
        if vis:
            index = 6
        img = self.coco.loadImgs(self.imgIds[index])[0]
        # img cv2 bgr to rgb
        img_file_path = osp.join(self.img_path, img['file_name'])
        input_img = np.ascontiguousarray(cv2.imread(img_file_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)[:,:,::-1])
        img_height, img_width = input_img.shape[:2]
        trans, inv_trans = gen_trans(img_width, img_height)

        input_img = cv2.warpAffine(input_img, trans, (cfg.input_img_shape[1], cfg.input_img_shape[0]), flags=cv2.INTER_LINEAR)        

        annIds = self.coco.getAnnIds(imgIds=img['id'], catIds=self.catIds, iscrowd=False)
        anns = self.coco.loadAnns(annIds)

        bboxes = [ann['bbox'] for ann in anns] #xywh
        bboxes = np.array(bboxes, dtype = np.float32)

        bboxes[:,2:] += bboxes[:,:2] #xywh -> xyxy
        nb = bboxes.shape[0]

        bboxes = bboxes.reshape(-1,2)
        bboxes_xy1 = np.concatenate([bboxes, np.ones((2*nb,1))],1)
        bboxes = np.dot(trans, bboxes_xy1.T).T[:,:2].reshape(nb,-1)


        ## bbox vis
        vis = False
        if vis:
            cvimg = copy.deepcopy(input_img)
            for bbox in bboxes:
                bbox_vis = copy.deepcopy(bbox)
                bbox_vis = bbox_vis.astype(int)
                cv2.rectangle(cvimg, bbox_vis[:2], bbox_vis[2:], (255,0,0), 3)
            cv2.imwrite('kihoon.png',cvimg)


        dummy_bbox = np.zeros((cfg.num_max_bbox-bboxes.shape[0], 4))
        num_valid_bbox = int(bboxes.shape[0])
        bboxes = np.concatenate([bboxes, dummy_bbox])

        input_img = self.transform(input_img)

        inputs = {'img': input_img}
        targets = {'bboxes': bboxes} ## xyxy
        meta_info = {'num_valid_bbox': num_valid_bbox, 'trans': trans, 'inv_trans': inv_trans}

        if(num_valid_bbox > 20):
            raise ValueError('bbox over 20 need more dummy padding')
        
        return inputs, targets, meta_info

    def __len__(self):
        return len(self.imgIds)
        
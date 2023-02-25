# MaskRCNN
Implementation of mask r-cnn for practice

# Week 0
Surveyed Related Papers

https://docs.google.com/presentation/d/1OyoITuqV_OD0Y37HTnhrMvYxfm3jYLLiMSrDXHctu_E/edit?usp=sharing

# Week 1
TODO:Implement dataloader of COCO, backbone models

02.14 implemented coco dataloader with img, bboxes

## GT_bboxes visualized
![Screenshot](/assets/gt_bboxes.png)

# Week 2
TODO:
- skeleton list up, outline
- dataloader augmentation(detectron2 ref.) +bbox transformation

02.21 Used Gyeongsik's codes as a reference and created baseline skeletons based on them

02.24
- Image preprocessing Implemented(resize to 1024**2, zeropadding, transformations)

## original image
![Alt text](/assets/orig.png?raw=true "Original Image")
## transformed image (1024x1024 zeropadded aspect_remained)
![Alt text](/assets/transformed.png?raw=true "Transformed Image")
## bbox also transformed properly under image preprocessing
![Screenshot](/assets/bbox_under_transformation.png)


- FPN and Anchor Generation code Implemented

## anchors under various feature levels
![Screenshot](/assets/4.png)
![Screenshot](/assets/8.png)
![Screenshot](/assets/16.png)
![Screenshot](/assets/32.png)
![Screenshot](/assets/64.png)


## whole_anchors
![Screenshot](/assets/whole_anchors.png)

02.25 TODO: IoU implementation, positive negative anchor implementation.
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

02.25 Implemented IoU(simple) function

# Week3

TODO:
- RPN bbox_reg, objectiveness head implementation

03.02 Implemented and Visualized Pos&Neg anchor(IoU > 0.5)
03.03 Implemented IoU tensor broadcasting instead of for loops

## positive anchors under various feature levels
![Screenshot](/assets/pos64.png)
![Screenshot](/assets/pos16.png)
![Screenshot](/assets/pos8.png)

03.05 implemented anchor_labeling, subsampling codes / visualized pos/neg windows

![Screenshot](/assets/pos_neg_1.png)
![Screenshot](/assets/pos_neg_2.png)
![Screenshot](/assets/pos_neg_3.png)

# Week4

03.10 Implemented rpn losses, transform_delta, RPN under train

# Week5

03.13 Tested rpn on one image

![Screenshot](/assets/vis_proposals/out10.png)
![Screenshot](/assets/vis_proposals/out20.png)
![Screenshot](/assets/vis_proposals/out50.png)
![Screenshot](/assets/vis_proposals/out100.png)
![Screenshot](/assets/vis_proposals/out200.png)

03.15 Implemented NMS function
03.21 Trained RPN, visualized qualitative results
![Screenshot](/assets/cherrypickings/1.png)
![Screenshot](/assets/cherrypickings/proposals_nms%20copy%202.png)
![Screenshot](/assets/cherrypickings/proposals_nms%20copy%203.png)
![Screenshot](/assets/cherrypickings/proposals_nms%20copy%204.png)
![Screenshot](/assets/cherrypickings/proposals_nms%20copy%205.png)
![Screenshot](/assets/cherrypickings/proposals_nms%20copy.png)
![Screenshot](/assets/cherrypickings/proposals_nms.png)

TODO: ROI align
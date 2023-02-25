import numpy

#xyxy
def area(box):
    return (box[2] - box[0]) * (box[3] - box[1])

#xyxy
def get_IoU(box1, box2):
    if (box1[0] <= box2[0]) and (box1[1] <= box2[1]):
        w_union = box1[2] - box2[0]
        h_union = box1[3] - box2[1]
    elif (box1[0] <= box2[0]) and (box1[1] > box2[1]):
        w_union = box1[2] - box2[0]
        h_union = box2[3] - box1[1]
    elif (box1[0] > box2[0]) and (box1[1] > box2[1]):
        w_union = box2[2] - box1[0]
        h_union = box2[3] - box1[1]
    elif (box1[0] > box2[0]) and (box1[1] <= box2[1]):
        w_union = box2[2] - box1[0]
        h_union = box1[3] - box2[1]
    else:
        raise ValueError('IoU imposible case')
    
    IoU = 0
    if ((w_union > 0) and (h_union > 0)):
        inter_area = w_union * h_union
        union_area = area(box1) + area(box2) - inter_area
        IoU = inter_area / union_area

    return IoU
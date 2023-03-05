import cv2
import numpy as np
from config import cfg

def gen_trans(orig_img_width, orig_img_height):
    # transformation from original image space to input image sapce
    src = np.zeros((3,2), dtype=np.float32)
    src[1] = np.array([0, orig_img_height-1], dtype=np.float32)
    src[2] = np.array([orig_img_width-1, 0], dtype=np.float32)

    dst = np.zeros((3,2), dtype=np.float32)

    if orig_img_width > orig_img_height:
        scale = cfg.input_img_shape[1]  / orig_img_width
        pad_x_left = 0
        pad_y_top = (cfg.input_img_shape[0] - orig_img_height*scale) // 2

    else:
        scale = cfg.input_img_shape[0]  / orig_img_height
        pad_x_left = (cfg.input_img_shape[1] - orig_img_width*scale) // 2
        pad_y_top = 0
    
    dst[0] = np.array([pad_x_left, pad_y_top], dtype=np.float32) #np float64 tensor transform problem
    dst[1] = np.array([pad_x_left, pad_y_top + scale*orig_img_height - 1], dtype=np.float32)
    dst[2] = np.array([pad_x_left + scale*orig_img_width - 1, pad_y_top], dtype=np.float32)

    inv_trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans, inv_trans


# import cv2

# def resize_pad_image(img, edge_length=1024):

#     # rescale max_edge to edge_length
#     _, h, w = img.shape
#     if h > w:
#         h_resized = edge_length
#         w_resized = edge_length * w/h
#         scale = edge_length / h
#     else:
#         h_resized = edge_length * h/w
#         w_resized = edge_length
#         sacle = edge_length / w
#     img_resized = cv2.resize(img, (w_resized, h_resized))

#     # pad square (edge_len x edge_len)
#     delta_w = edge_length - w_resized
#     delta_h = edge_length - h_resized

#     top, bottom = delta_h//2, delta_h-(delta_h//2)
#     left, right = delta_w//2, delta_w-(delta_w//2)
#     img_resize_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0,0,0])

#     return img_resize_padded, scale
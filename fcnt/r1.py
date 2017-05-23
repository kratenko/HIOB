"""
Created on 2016-09-06

@author: Peer Springstübe
"""

from PIL import Image
import math

fpath = "/data/Peer/data/tb100_unzipped/SHORT/img/0001.jpg"

# def ext_roi(im, GT, l_off, roi_size, r_w_scale):


def calc_scale(pos):
    roi_scale = [2.0, 2.0]
    w = pos[2]
    h = pos[3]
    dia = math.sqrt(w * w + h * h)
    print("dia", dia)
    scale = [dia / w, dia / h]
    return scale[0] * roi_scale[0], scale[1] * roi_scale[1]


def ext_roi(img, last_pos, l_off, roi_size, r_w_scale):
    # size of whole image
    i_w, i_h = img.size
    # size and position of position
    p_x, p_y, p_w, p_h = last_pos
    #
    p_cx = round(p_x + p_w / 2 + l_off[0])
    p_cy = round(p_y + p_h / 2 + l_off[1])
    print("p_c", p_cx, p_cy)
    #
    roi_w = r_w_scale[0] * p_w
    roi_h = r_w_scale[1] * p_h
    print("roi", roi_w, roi_h)
    #
    x1 = p_cx - round(roi_w / 2)
    y1 = p_cy - round(roi_h / 2)
    x2 = p_cx + round(roi_w / 2)
    y2 = p_cy + round(roi_h / 2)
    print("xy", x1, y1, x2, y2)
    #
    clip = min([x1, y1, i_h - y2, i_w - x2])
    print("clip", clip)
    #
    pad = 0
    if clip <= 0:
        pad = abs(clip) + 1
        x1 += pad
        x2 += pad
        y1 += pad
        y2 += pad
    print("pad", pad)
    roi_pos = [x1, y1, x2 - x1 + 1, y2 - y1 + 1]
    roi_rect = [x1, y1, x2 + 1, y2 + 1]
    print("roi_pos", roi_pos)
    #
    roi_img = img.crop(box=roi_rect).resize((roi_size, roi_size))
    return roi_img, None, roi_pos, pad


def calc_roi(img_size, pos, offset=None):
    if offset is None:
        offset = [0, 0]
    # size of whole image
    i_w, i_h = img_size
    # size and position of position
    p_x, p_y, p_w, p_h = pos
    #
    p_cx = round(p_x + p_w / 2 + offset[0])
    p_cy = round(p_y + p_h / 2 + offset[1])
    print("p_c", p_cx, p_cy)
    #
    r_w_scale = calc_scale(pos)
    roi_w = r_w_scale[0] * p_w
    roi_h = r_w_scale[1] * p_h
    print("roi", roi_w, roi_h)
    #
    x1 = p_cx - round(roi_w / 2)
    y1 = p_cy - round(roi_h / 2)
    x2 = p_cx + round(roi_w / 2)
    y2 = p_cy + round(roi_h / 2)
    print("xy", x1, y1, x2, y2)
    #
    clip = min([x1, y1, i_h - y2, i_w - x2])
    print("clip", clip)
    #
    pad = 0
    if clip <= 0:
        pad = abs(clip) + 1
        x1 += pad
        x2 += pad
        y1 += pad
        y2 += pad
    print("pad", pad)
    roi_pos = [x1, y1, x2 - x1 + 1, y2 - y1 + 1]
    roi_rect = [x1, y1, x2 + 1, y2 + 1]
    print("roi_pos", roi_pos)
    #
    roi_img = img.crop(box=roi_rect).resize((roi_size, roi_size))
    return roi_img, None, roi_pos, pad


img = Image.open(fpath)

pos = [269, 75, 34, 64]
scale = calc_scale(pos)
print("size", img.size)
print("pos", pos)
print("scale", scale)
roi_img, _, roi_pos, pad = ext_roi(img, pos, [0, 0], 368, scale)
roi_img.show()

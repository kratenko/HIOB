"""
Created on 2016-11-03

@author: Peer Springstübe
"""
import math

from hiob.rect import Rect


class Roi(object):
    pass


class RoiExtractor(object):

    def __init__(self):
        pass

    def extract_roi(self, image, position):
        return Rect(0, 0, 0, 0)


class FcntRoiCalculator(object):

    def calculate_scale(self, position):
        # magic number taken from init_tracker.m, pf_param.roi_scale
        roi_scale = [2.0, 2.0]
        w = position.width
        h = position.height
        # fcnt only uses initial frame's size for this
        dia = math.sqrt(w * w + h * h)
        scale = [dia / w, dia / h]
        return scale[0] * roi_scale[0], scale[1] * roi_scale[1]

    def calculate_roi(self, image, position):
        i_w, i_h = image.size

        c_x, c_y = position.center
        r_w_scale = self.calculate_scale(position)
        roi_w = r_w_scale[0] * position.width
        roi_h = r_w_scale[1] * position.height

        x1 = c_x - round(roi_w / 2)
        y1 = c_y - round(roi_h / 2)
        x2 = c_x + round(roi_w / 2)
        y2 = c_y + round(roi_h / 2)

        clip = min([x1, y1, i_w - x2, i_h - y2])
        pad = 0
        if clip <= 0:
            pad = abs(clip) + 1
            x1 += pad
            y1 += pad
            x2 += pad
            y2 += pad
        roi = Rect(x1, y1, x2 - x1 + 1, y2 - y1 + 1)
        return roi


class SimpleRoiCalculator(object):

    def calculate_scale(self, position):
        # magic number taken from init_tracker.m, pf_param.roi_scale
        roi_scale = [2.0, 2.0]
        w = position.width
        h = position.height
        # fcnt only uses initial frame's size for this
        dia = math.sqrt(w * w + h * h)
        scale = [dia / w, dia / h]
        return scale[0] * roi_scale[0], scale[1] * roi_scale[1]

    def calculate_roi(self, image, position):
        i_w, i_h = image.size

        c_x, c_y = position.center
        r_w_scale = self.calculate_scale(position)
        roi_w = r_w_scale[0] * position.width
        roi_h = r_w_scale[1] * position.height

        # only one size, since we want a square
        s = round((roi_w + roi_h) / 2) + 1
        # make sure it fits in the image:
        s = min(s, i_w - 1, i_h - 1)

        # fit roi on upper left side:
        x1 = max(c_x - s // 2, 0)
        y1 = max(c_y - s // 2, 0)
        # fit roi on bottom right side:
        if x1 + s >= i_w:
            x1 = i_w - s - 1
        if y1 + s >= i_h:
            y1 = i_h - s - 1
        return Rect(x1, y1, s, s)

        x2 = x1 + s
        y2 = y1 + s
        if x2 >= i_w:
            pad_x = x2 - i_w + 1
            x1 -= pad_x
            x2 -= pad_x
        if y2 >= i_h:
            pad_y = y2 - i_h + 1
            y1 -= pad_y
            y2 -= pad_y

        x1 = c_x - round(roi_w / 2)
        y1 = c_y - round(roi_h / 2)
        x2 = c_x + round(roi_w / 2)
        y2 = c_y + round(roi_h / 2)

        clip = min([x1, y1, i_w - x2, i_h - y2])
        pad = 0
        if False and clip <= 0:
            pad = abs(clip) + 1
            x1 += pad
            y1 += pad
            x2 += pad
            y2 += pad
        roi = Rect(x1, y1, x2 - x1 + 1, y2 - y1 + 1)
        return roi

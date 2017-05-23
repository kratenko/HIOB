"""
Created on 2016-10-06

@author: Peer Springstübe
"""
import math
import logging
import numpy as np

from hiob.rect import Rect

logger = logging.getLogger(__name__)


def _calc_scale(pos):
    roi_scale = [2.0, 2.0]
    w = pos[2]
    h = pos[3]
    dia = math.sqrt(w * w + h * h)
    logger.debug("dia: %f", dia)
    # print("dia", dia)
    scale = [dia / w, dia / h]
    return scale[0] * roi_scale[0], scale[1] * roi_scale[1]


def _calculate_roi(img_size, pos, offset=None):
    """
    Calculate ROI from image size and current position.
    FCNT: utils/ext_roi
    """
    if offset is None:
        offset = [0, 0]
    # size of whole image
    i_w, i_h = img_size
    # size and position of position
    p_x, p_y, p_w, p_h = pos
    #
    p_cx = round(p_x + p_w / 2 + offset[0])
    p_cy = round(p_y + p_h / 2 + offset[1])
    # print("p_c", p_cx, p_cy)
    #
    r_w_scale = _calc_scale(pos)
    roi_w = r_w_scale[0] * p_w
    roi_h = r_w_scale[1] * p_h
    # print("roi", roi_w, roi_h)
    #
    x1 = p_cx - round(roi_w / 2)
    y1 = p_cy - round(roi_h / 2)
    x2 = p_cx + round(roi_w / 2)
    y2 = p_cy + round(roi_h / 2)
    # print("xy", x1, y1, x2, y2)
    #
    clip = min([x1, y1, i_h - y2, i_w - x2])
    # print("clip", clip)
    #
    pad = 0
    if clip <= 0:
        pad = abs(clip) + 1
        x1 += pad
        x2 += pad
        y1 += pad
        y2 += pad
    # print("pad", pad)
    w = x2 - x1 + 1
    h = y2 - y1 + 1
    roi_pos = [x1, y1, w, h]
    roi_rect = [x1, y1, x2 + 1, y2 + 1]
    # print("roi_pos", roi_pos)
    # logger.info("ROI: img_size=(%d,%d), pos=(%d,%d,%d,%d), roi=(%d,%d,%d,%d)",
    #            i_w, i_h, p_x, p_y, p_w, p_h, x1, y2, w, h)
    return roi_pos, roi_rect


class Frame(object):
    """
    Class representing a single frame of the sequence.
    """
    STD_INPUT_SIZE = [368, 368]

    _INSTANCE_COUNTER = 0
    instance_id = None

    input_size = None

    img_rect = None
    roi_rect = None
    pos_rect = None
    prev_pos_rect = None

    img = None
    roi_img = None
    input_img = None
    feature_input = None

    def __init__(self, img, input_size=None):
        Frame._INSTANCE_COUNTER += 1
        self.instance_id = Frame._INSTANCE_COUNTER
        logger.debug("construct <Frame#%d>", self.instance_id)

        # Store frame image and extract size as rect:
        self.img = img
        self.img_rect = Rect(img.size)
        # Size of network input
        if input_size is None:
            self.input_size = Frame.STD_INPUT_SIZE
        else:
            self.input_size = input_size

    def __repr__(self):
        return "<Frame#{}>".format(self.instance_id)

    def calculate_roi_rect(self):
        """
        Calculate the ROI for this frame.

        Calculate the region of interest for the current frame, 
        using the position of the tracked object in the previous frame.
        """
        logger.debug("calculate ROI for %s", self)
        if self.prev_pos_rect is None:
            raise ValueError(
                "cannot calculate ROI if prev_pos_rect is not set.")
        roi_pos, _ = _calculate_roi(
            self.img_rect.size, self.prev_pos_rect, None)
        self.roi_rect = Rect(roi_pos)
        logger.info(
            "ROI information for %s: img_size=(%d,%d), prev_pos=(%d,%d,%d,%d), roi=(%d,%d,%d,%d)",
            self, *(self.img_rect.size + self.prev_pos_rect.tuple + self.roi_rect.tuple))

    def query_roi_rect(self):
        if self.roi_rect is None:
            self.calculate_roi_rect()
        return self.roi_rect

    def calculate_roi_img(self, size):
        self.roi_img = self.img.crop(box=self.roi_rect.outer).resize(size)

    # def query_roi_img(self):
    #    if self.roi_img is None:
    #        self.calculate_roi_image()
    #    return self.roi_img


"""
raw
roi in raw
roi scaled for fex
feat
"""

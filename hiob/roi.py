"""
Created on 2016-11-17

@author: Peer Springstübe
"""

import math

from hiob.base import HiobModule
from hiob.rect import Rect


class RoiCalculator(HiobModule):

    def calculate_roi(self, frame):
        raise NotImplementedError()


class SimpleRoiCalculator(RoiCalculator):

    def configure(self, configuration):
        # magic number taken from init_tracker.m, pf_param.roi_scale
        self.roi_scale = [2.0, 2.0]

    def calculate_scale(self, position):
        # magic number taken from init_tracker.m, pf_param.roi_scale
        w = position.width
        h = position.height
        # fcnt only uses initial frame's size for this
        dia = math.sqrt(w * w + h * h)
        scale = [dia / w, dia / h]
        return scale[0] * self.roi_scale[0], scale[1] * self.roi_scale[1]

    def calculate_roi(self, frame):
        i_w, i_h = frame.capture_image.size

        position = frame.previous_position

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

        # set roi in frame:
        frame.roi = Rect(x1, y1, s, s)

    def setup(self, session):
        pass


class SroiGenerator(HiobModule):

    def generate_sroi(self, frame):
        raise NotImplementedError()


class SimpleSroiGenerator(SroiGenerator):

    def configure(self, configuration):
        self.sroi_size = configuration['sroi_size']

    def setup(self, session):
        pass

    def generate_sroi(self, frame):
        frame.sroi_image = frame.capture_image.crop(
            frame.roi.outer).resize(self.sroi_size)

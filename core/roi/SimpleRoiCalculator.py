"""
Created on 2016-11-17

@author: Peer Springstübe
"""

import math

from ..Rect import Rect
from .RoiCalculator import RoiCalculator


class SimpleRoiCalculator(RoiCalculator):

    def configure(self, configuration):
        # magic number taken from init_tracker.m, pf_param.roi_scale
        self.roi_scale = configuration["roi_scale"] if "roi_scale" in configuration else [2.0, 2.0]
        self.fixed_size = configuration["roi_fixed_size"] if "roi_fixed_size" in configuration else None
        self.roi_movement_factor = configuration["roi_movement_factor"] if "roi_movement_factor" in configuration\
            else 1.0
        self.old_size_calculation = configuration['old_size_calculation'] if 'old_size_calculation' in configuration\
            else False

    def set_initial_position(self, position):
        self.initial_position = position

    def calculate_scale(self, position, previous_position):
        # magic number taken from init_tracker.m, pf_param.roi_scale
        w = position.width
        h = position.height
        # fcnt only uses initial frame's size for this
        dia = math.sqrt(w * w + h * h)
        scale = [dia / w, dia / h]
        return scale[0] * self.roi_scale[0], scale[1] * self.roi_scale[1]

    def calculate_size(self, position, previous_position):
        # magic number taken from init_tracker.m, pf_param.roi_scale
        # fcnt only uses initial frame's size for this
        #dia = math.sqrt(position.width ** 2 + position.height ** 2)
        #scale = [dia / w, dia / h]
        #return scale[0] * self.roi_scale[0], scale[1] * self.roi_scale[1]

        base_scale = position.center_distance(previous_position) * 2 * self.roi_movement_factor
        if self.fixed_size:
            return self.fixed_size[0] + self.roi_movement_factor * self.roi_scale[0], \
                   self.fixed_size[1] + self.roi_movement_factor * self.roi_scale[1]
        else:
            return (max(self.initial_position.width, position.width) * 1.5 + base_scale) * self.roi_scale[0], \
                   (max(self.initial_position.height, position.height) * 1.5 + base_scale) * self.roi_scale[1]

    def calculate_roi(self, frame):
        i_w, i_h = frame.size

        position = frame.previous_position

        c_x, c_y = position.center
        # original implementation:
        if self.old_size_calculation:
            r_w_scale = self.calculate_scale(position, frame.before_previous_position)
            roi_w = r_w_scale[0] * position.width
            roi_h = r_w_scale[1] * position.height
        else:

            if self.fixed_size == "full":
                roi_w = roi_h = min(frame.size)
            else:
                roi_w, roi_h = self.calculate_size(position, frame.before_previous_position)
        print("roi size: {}|{}".format(roi_w, roi_h))
        #print("roi height: {}|{}".format(roi_h, roi_h1))

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

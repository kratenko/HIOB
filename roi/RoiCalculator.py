"""
Created on 2016-11-17

@author: Peer Springstübe
"""


from hiob.HiobModule import HiobModule


class RoiCalculator(HiobModule):

    def calculate_roi(self, frame):
        raise NotImplementedError()
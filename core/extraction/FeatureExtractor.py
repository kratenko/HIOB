"""
Created on 2016-11-17

@author: Peer Springstübe
"""

from .. import HiobModule



class FeatureExtractor(HiobModule.HiobModule):

    def extract_features(self, tracking, frame):
        raise NotImplementedError()



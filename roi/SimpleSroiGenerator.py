"""
Created on 2016-11-17

@author: Peer Springstübe
"""


from .SroiGenerator import SroiGenerator


class SimpleSroiGenerator(SroiGenerator):

    def configure(self, configuration):
        self.sroi_size = configuration['sroi_size']

    def setup(self, session):
        pass

    def generate_sroi(self, frame):
        frame.sroi_image = frame.capture_image.crop(
            frame.roi.outer).resize(self.sroi_size)
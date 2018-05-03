"""
Created on 2016-11-17

@author: Peer Springstübe
"""


from .SroiGenerator import SroiGenerator
from PIL import Image
import numpy as np
import tensorflow as tf
import time
import zipfile, io, re


class SimpleSroiGenerator(SroiGenerator):

    def configure(self, configuration):
        self.sroi_size = configuration['sroi_size']
        self.resize_on_gpu = configuration['sroi_gpu_resize'] if 'gpu_resize' in configuration else True

    def setup(self, session):
        pass

    def generate_sroi(self, frame):

        if self.resize_on_gpu:
            im = frame.capture_image
            tf_img = tf.convert_to_tensor(np.expand_dims(np.array(im), axis=0) / 255)
            bbox = (
                frame.roi.y / im.size[1] ,
                frame.roi.x / im.size[0],
                (frame.roi.y + frame.roi.height) / im.size[1],
                (frame.roi.x + frame.roi.width) / im.size[0])
            bbox_tensor = tf.reshape(tf.convert_to_tensor(bbox), [1, 4])

            resized = tf.image.crop_and_resize(
                tf_img,
                bbox_tensor,
                [0],
                self.sroi_size)
            frame.sroi_image = Image.fromarray(np.asarray((resized.eval() * 255)[0], dtype=np.uint8))
        else:
            frame.sroi_image = frame.capture_image.crop(
                frame.roi.outer).resize(self.sroi_size, Image.LANCZOS)


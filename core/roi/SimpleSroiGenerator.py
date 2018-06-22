"""
Created on 2016-11-17

@author: Peer Springstübe
"""


from .SroiGenerator import SroiGenerator
from PIL import Image
import numpy as np
import tensorflow as tf
import time
import logging

logger = logging.getLogger(__name__)


class SimpleSroiGenerator(SroiGenerator):

    def __init__(self):
        self.sroi_size = None
        self.resize_on_gpu = True
        self.session = None

        self.generated_sroi = None
        self.generated_sroi_tensor = None
        self.bbox_placeholder = None
        self.input_placeholder = None
        self.cache_sroi = None

    def configure(self, configuration):
        self.sroi_size = configuration['sroi_size']
        self.resize_on_gpu = configuration['sroi_gpu_resize'] if 'gpu_resize' in configuration else True

    def setup(self, session, size):
        self.session = session
        self.build_tf_model(size)

    def generate_sroi(self, frame):

        the_bbox = self.get_bbox(frame)
        self.session.run(
            [self.cache_sroi],
            feed_dict={
                self.input_placeholder: frame.capture_image,
                self.bbox_placeholder: the_bbox
            })

    def get_bbox(self, frame):
        bbox = np.array([
            frame.roi.y / frame.size[1],
            frame.roi.x / frame.size[0],
            (frame.roi.y + frame.roi.height) / frame.size[1],
            (frame.roi.x + frame.roi.width) / frame.size[0]])
        return bbox

    def build_tf_model(self, shape):
        logger.info("build model started")
        start_time = time.time()
        self.input_placeholder = tf.placeholder(
            dtype=tf.uint8,
            shape=[shape[1], shape[0], 3],
            name='input_placeholder')
        im = tf.reshape(self.input_placeholder, [1, shape[1], shape[0], 3])

        self.bbox_placeholder = tf.placeholder(
            dtype=tf.float32, shape=[4], name='bbox_ph')
        bbox = tf.reshape(self.bbox_placeholder, [1, 4])

        generated_sroi_tensor = tf.image.crop_and_resize(
            im,
            bbox,
            [0],
            self.sroi_size)
        self.generated_sroi = tf.get_variable('sroi',
                                              shape=(1, self.sroi_size[1], self.sroi_size[0], 3),
                                              initializer=tf.zeros_initializer())
        self.cache_sroi = self.generated_sroi.assign(generated_sroi_tensor)
        logger.info("build model finished %ds", time.time() - start_time)




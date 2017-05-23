"""
Created on 2016-11-17

@author: Peer Springstübe
"""

import logging

import numpy as np
import tensorflow as tf

import hiob.base
from hiob import vgg16
from hiob import alexnet
from collections import OrderedDict
import os

logger = logging.getLogger(__name__)


class FeatureExtractor(hiob.base.HiobModule):

    def extract_features(self, tracking, frame):
        raise NotImplementedError()


class CnnFeatureExtractor(FeatureExtractor):

    def configure(self, configuration):
        self.configuration = configuration
        self.sroi_size = configuration['sroi_size']
        self.input_shape = [1, self.sroi_size[0], self.sroi_size[1], 3]
        self.net_dir = configuration['net_dir']
        self.net_name = configuration['extractor_net']
        self.requested_feature_names = [x[0]
                                        for x in configuration['features']]
        self.output_size = configuration['mask_size']

    def setup(self, session):
        self.session = session
        if self.net_name == 'vgg16':
            logger.info("creating pretrained vgg16 net as feature extractor")
            net_path = os.path.join(self.net_dir, 'vgg16.npy')
            self.net = vgg16.Vgg16(
                input_size=self.sroi_size,
                vgg16_npy_path=net_path,
            )
        elif self.net_name == 'alexnet':
            logger.info("creating pretrained alexnet as feature extractor")
            net_path = os.path.join(self.net_dir, 'alexnet.npy')
            self.net = alexnet.AlexNet(
                input_size=self.sroi_size,
                alexnet_npy_path=net_path,
            )
        else:
            raise ValueError(
                "Can only use vgg16 and alexnet, yet, not %s" % self.net_name)

        self.requested_features = OrderedDict()
        for n in self.requested_feature_names:
            self.requested_features[n] = self.net.features[n]

        if self.output_size is None:
            # find biggest shape in output features:
            self.output_size = [0, 0]
            for f in self.requested_features.values():
                sz = f.get_shape().as_list()[1:3]
                if sz[0] > self.output_size[0]:
                    self.output_size[0] = sz[0]
                if sz[1] > self.output_size[1]:
                    self.output_size[1] = sz[1]

        # build feature outputs:
        self.output_features = OrderedDict()
        for n, f in self.requested_features.items():
            sz = f.get_shape().as_list()[1:3]
            if sz == self.output_size:
                # is right shape, just use feature directly:
                self.output_features[n] = f
            else:
                # feature has wrong size, resize:
                resizer = tf.image.resize_images(f, self.output_size)
                self.output_features[n] = resizer

    def extract_features(self, tracking, frame):
        logger.info("Extracting features for %s", frame)
        # get np-array from image data as input:
        feature_input = np.array(frame.sroi_image.getdata(), dtype=np.float32).reshape(
            self.input_shape)
        # execute net to get requested output layers:
        outputs = self.session.run(
            list(self.output_features.values()),
            feed_dict={self.net.input_placeholder: feature_input})
        # pack for frame storage:
        features = OrderedDict()
        for n, o in enumerate(outputs):
            features[self.requested_feature_names[n]] = o
        # store features in frame:
        frame.features = features

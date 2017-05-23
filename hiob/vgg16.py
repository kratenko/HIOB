import inspect
import os
import logging

import numpy as np
import tensorflow as tf
import time
from collections import OrderedDict

logger = logging.getLogger(__name__)


class Vgg16(object):
    VGG_MEAN = [103.939, 116.779, 123.68]

    def __init__(self, input_size=None, vgg16_npy_path=None):
        if input_size is None:
            input_size = (224, 224)

        if vgg16_npy_path is None:
            path = inspect.getfile(Vgg16)
            path = os.path.abspath(os.path.join(path, os.pardir))
            path = os.path.join(path, "vgg16.npy")
            vgg16_npy_path = path

        logger.info("VGG16 file: '%s'", vgg16_npy_path)

        self.input_size = input_size
        self.input_shape = (1, input_size[0], input_size[1], 3)

        self.input_placeholder = tf.placeholder(
            dtype=tf.float32, shape=self.input_shape, name='input_placeholder')

        # convert from rgb to bgr and apply mean
        red, green, blue = tf.split(3, 3, self.input_placeholder)
        assert red.get_shape().as_list()[1:] == [
            self.input_size[0], self.input_size[1], 1]
        assert green.get_shape().as_list()[1:] == [
            self.input_size[0], self.input_size[1], 1]
        assert blue.get_shape().as_list()[1:] == [
            self.input_size[0], self.input_size[1], 1]
        bgr = tf.concat(3, [
            blue - self.VGG_MEAN[0],
            green - self.VGG_MEAN[1],
            red - self.VGG_MEAN[2],
        ])
        assert bgr.get_shape().as_list()[1:] == [
            self.input_size[0], self.input_size[1], 3]

        self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
        logger.info("npy file loaded")

        self.build()

    def build(self):
        """
        load variable from npy to build the VGG

        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        """

        start_time = time.time()
        logger.info("build model started")
        # rgb_scaled = rgb * 255.0

        # Convert RGB to BGR
        red, green, blue = tf.split(3, 3, self.input_placeholder)
        assert red.get_shape().as_list()[1:] == [
            self.input_size[0], self.input_size[1], 1]
        assert green.get_shape().as_list()[1:] == [
            self.input_size[0], self.input_size[1], 1]
        assert blue.get_shape().as_list()[1:] == [
            self.input_size[0], self.input_size[1], 1]
        bgr = tf.concat(3, [
            blue - self.VGG_MEAN[0],
            green - self.VGG_MEAN[1],
            red - self.VGG_MEAN[2],
        ])
        assert bgr.get_shape().as_list()[1:] == [
            self.input_size[0], self.input_size[1], 3]
        assert bgr.dtype == tf.float32

        self.conv1_1 = self.conv_layer(bgr, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
        self.pool3 = self.max_pool(self.conv3_3, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
        self.pool4 = self.max_pool(self.conv4_3, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
        self.pool5 = self.max_pool(self.conv5_3, 'pool5')

        # store feature layers:
        f = OrderedDict()
        f['conv1_1'] = self.conv1_1
        f['conv1_2'] = self.conv1_2
        f['pool1'] = self.pool1
        f['conv2_1'] = self.conv2_1
        f['conv2_2'] = self.conv2_2
        f['pool2'] = self.pool2
        f['conv3_1'] = self.conv3_1
        f['conv3_2'] = self.conv3_2
        f['conv3_3'] = self.conv3_3
        f['pool3'] = self.pool3
        f['conv4_1'] = self.conv4_1
        f['conv4_2'] = self.conv4_2
        f['conv4_3'] = self.conv4_3
        f['pool4'] = self.pool4
        f['conv5_1'] = self.conv5_1
        f['conv5_2'] = self.conv5_2
        f['conv5_3'] = self.conv5_3
        f['pool5'] = self.pool5
        self.features = f

        # free init data:
        del self.data_dict

        logger.info("build model finished: %ds", (time.time() - start_time))

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    def get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name="filter")

    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name="biases")

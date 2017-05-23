"""
Created on 2016-09-06

@author: Peer Springstübe
"""

import tensorflow as tf
import numpy as np

VGG_MEAN = [103.939, 116.779, 123.68]

vgg_size = 2


rgb_input = tf.placeholder(
    tf.float32, shape=(1, vgg_size, vgg_size, 3), name="rgb_input")
with tf.name_scope('preprocess'):
    red, green, blue = tf.split(3, 3, rgb_input)
    bgr = tf.concat(3, [
        blue - VGG_MEAN[0],
        green - VGG_MEAN[1],
        red - VGG_MEAN[2],
    ])

vgg16f_input = tf.placeholder(
    tf.float32, shape=(1, vgg_size, vgg_size, 3), name="vgg16f_input")


data = np.array(
    range(vgg_size * vgg_size * 3)).reshape(1, vgg_size, vgg_size, 3)

print(data)

with tf.Session() as sess:
    out = sess.run(bgr, feed_dict={rgb_input: data})
    print(out)
    print(sess.graph_def)

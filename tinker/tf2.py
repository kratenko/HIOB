import math

import numpy as np
import tensorflow as tf
from PIL import Image

fpath = "/data/Peer/data/tb100_unzipped/SHORT/img/0001.jpg"
img = Image.open(fpath)


def calc_scale(pos):
    roi_scale = [2.0, 2.0]
    w = pos[2]
    h = pos[3]
    dia = math.sqrt(w * w + h * h)
    scale = [dia / w, dia / h]
    return scale[0] * roi_scale[0], scale[1] * roi_scale[1]

dtype = tf.float32

# image as np-array
ir = np.array(img.getdata()).reshape((img.size[0], img.size[1], 3))

pos = tf.placeholder(dtype, (4,))
dia = (pos[2]**2 + pos[3]**2)**.5
scale = [dia / pos[2] * 2.0, dia / pos[3] * 2.0]

with tf.name_scope('ext_roi'):
    image_raw = tf.placeholder(tf.int8, (None, None, 3), 'image_raw')
    image_size = tf.placeholder(tf.int8, (2,), 'image_size')
    pos_cx = tf.round(pos[0] + pos[2] / 2 + 0)
    pos_cy = tf.round(pos[1] + pos[3] / 2 + 0)
    roi_w = scale[0] * pos[2]
    roi_h = scale[1] * pos[3]
    x1 = pos_cx - roi_w // 2
    y1 = pos_cy - roi_h // 2
    x2 = pos_cx + roi_w // 2
    y2 = pos_cy + roi_h // 2
    #
    clip = tf.minimum(tf.minimum(x1, x2), tf.minimum(y1, y2))
    roi_pos = [x1, y1, x2 - x1 + 1, y2 - y1 + 1]
    roi_rect = [x1, y1, x2 + 1, y2 + 1]


with tf.Session() as sess:
    out = sess.run(
        [roi_pos, roi_rect], feed_dict={pos: [269, 75, 34, 64], image_raw: ir, image_size: img.size})
    print(out, img.size)

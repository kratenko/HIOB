"""
Created on 2016-10-06

@author: Peer Springstübe
"""

import math
import re
import logging
from PIL import Image, ImageDraw
from hiob.rect import Rect
from hiob.frame import Frame
from hiob.tracker import Tracker
import matplotlib.pyplot as plt
from scipy.io import loadmat
import tensorflow as tf
import numpy as np
from hiob.data_set import load_tb100_sample

logging.basicConfig(
    level=logging.DEBUG, format='[%(asctime)s|%(name)s|%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

sample = "MotorRolling"

p = "/data/Peer/data/tb100_unzipped/" + sample
fpath = p + "/img/0001.jpg"
lpath = p + "/lid.mat"
gpath = p + "/gid.mat"
gtpath = p + "/groundtruth_rect.txt"
lid = loadmat(lpath)['lid'].reshape(512)
gid = loadmat(gpath)['gid'].reshape(512)


def _groundtruth_line(line):
    r = re.compile(r"(\d+)\D+(\d+)\D+(\d+)\D+(\d+)")
    m = r.match(str(line))
    if m:
        return tuple(int(m.group(n)) for n in (1, 2, 3, 4))
    else:
        return None


with open(gtpath, 'r') as gtf:
    for line in gtf:
        gt = _groundtruth_line(line)
        break


img = Image.open(fpath)

frame = Frame(img)

ds = load_tb100_sample(sample)

tr = Tracker()
tr.initial_position = Rect(gt)
tr.generate_feature_input(frame)
frame.roi_img.show()
#frame.feature_input /= 256
# print(frame.feature_input)
# plt.imshow(frame.feature_input[0])
# plt.show()

tr.feature_order = {
    'conv4': lid, 'conv5': gid}

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    tr.sess = sess
    tr.build_net()

    for n, im in enumerate(ds.images):
        logger.info("Frame %d", n)
        fr = Frame(im)
        tr.generate_feature_input(fr)
        tr.generate_features(fr)
        tr.reduce_features(fr, 16)


c4 = frame.reduced_features['conv4']
c5 = frame.reduced_features['conv5']
print(c4.shape, c5.shape)

f = c5

rot = np.swapaxes(np.swapaxes(f[0], 0, 2), 1, 2)

for prep in rot:
    #prep = np.swapaxes(np.swapaxes(f[0], 0, 2), 1, 2)[0]
    plt.imshow(prep, cmap='hot', interpolation='nearest')
    plt.show()

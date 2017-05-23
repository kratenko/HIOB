import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
from PIL import Image, ImageDraw

from fcnt import vgg16, gauss

fpath = "/data/Peer/data/tb100_unzipped/SHORT/img/0001.jpg"
#fpath = "/data/Peer/data/tb100_unzipped/MotorRolling/img/0001.jpg"
#fpath = '/data/Peer/Jogging/img/0027.jpg'


def calc_scale(pos):
    roi_scale = [2.0, 2.0]
    w = pos[2]
    h = pos[3]
    dia = math.sqrt(w * w + h * h)
    print("dia", dia)
    scale = [dia / w, dia / h]
    return scale[0] * roi_scale[0], scale[1] * roi_scale[1]


def calc_roi(img_size, pos, offset=None):
    if offset is None:
        offset = [0, 0]
    # size of whole image
    i_w, i_h = img_size
    # size and position of position
    p_x, p_y, p_w, p_h = pos
    #
    p_cx = round(p_x + p_w / 2 + offset[0])
    p_cy = round(p_y + p_h / 2 + offset[1])
    print("p_c", p_cx, p_cy)
    #
    r_w_scale = calc_scale(pos)
    roi_w = r_w_scale[0] * p_w
    roi_h = r_w_scale[1] * p_h
    print("roi", roi_w, roi_h)
    #
    x1 = p_cx - round(roi_w / 2)
    y1 = p_cy - round(roi_h / 2)
    x2 = p_cx + round(roi_w / 2)
    y2 = p_cy + round(roi_h / 2)
    print("xy", x1, y1, x2, y2)
    #
    clip = min([x1, y1, i_h - y2, i_w - x2])
    print("clip", clip)
    #
    pad = 0
    if clip <= 0:
        pad = abs(clip) + 1
        x1 += pad
        x2 += pad
        y1 += pad
        y2 += pad
    print("pad", pad)
    roi_pos = [x1, y1, x2 - x1 + 1, y2 - y1 + 1]
    roi_rect = [x1, y1, x2 + 1, y2 + 1]
    print("roi_pos", roi_pos)
    return roi_pos, roi_rect


# 46x46x512
# dropout ratio=0.3
# conv, 3x3x1, pad=1; weight_dea
"""
layers {
  bottom: "conv5_f0"
 # bottom: "data"
  top: "conv5_f1"
  name: "conv5_f1"
  type: CONVOLUTION
  blobs_lr:1 # 30# 15
  blobs_lr:2 # 40# 30
  weight_decay: 2 #1
  weight_decay: 0 #0
  convolution_param {
    num_output: 1
    pad: 1
    kernel_size: 3
    weight_filler {
      type: "gaussian"
      std: 1e-7
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
"""


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def bias_variable(shape, name=None):
    # initial = tf.constant(, shape=shape)
    initial = tf.zeros(shape)
    return tf.Variable(initial, name)


def weight_variable(shape, name=None):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)


def build_fit_net():
    feat_w, feat_h = 46, 46
    feat_num = 512
    pass


class CnnSelNet(object):
    feature_size = 46, 46
    feature_number = 512
    kernel_size = 3
    dropout_keep = 0.7
    dtype = tf.float32
    learning_rate = 0.01

    def __init__(self, name):
        self.name = name
        with tf.name_scope(name):
            self.build_net()
            self.build_trainer()

    def build_net(self):
        # input vaules:
        self.input_shape = (
            1, self.feature_size[0], self.feature_size[1], self.feature_number)
        self.input = tf.placeholder(self.dtype, self.input_shape, name="input")
        # dropout layer:
        self.dropout = tf.nn.dropout(
            self.input, self.dropout_keep, name="dropout")
        # conv-layer
        self.conv_shape = (
            self.kernel_size, self.kernel_size, self.feature_number, 1)
        conv_weight_initial = tf.truncated_normal(self.conv_shape, stddev=0.1)
        self.conv_weight = tf.Variable(conv_weight_initial, name="conv_weight")
        self.conv_bias_shape = [1]
        self.conv_bias = bias_variable(self.conv_bias_shape, name='conv_bias')
        self.conv = tf.nn.conv2d(self.dropout, self.conv_weight, strides=[
                                 1, 1, 1, 1], padding='SAME', name="conv") + self.conv_bias

    def build_trainer(self):
        # input for gt:
        self.truth_input_shape = (
            1, self.feature_size[0], self.feature_size[1], 1)
        self.truth_input = tf.placeholder(
            self.dtype, self.truth_input_shape, "truth_input")
        # cost function
        self.cost = tf.reduce_mean(tf.square(self.conv - self.truth_input))
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train = self.optimizer.minimize(self.cost)


vgg_size = 368
feat_size = 46

img = Image.open(fpath)
#img.resize((vgg_size, vgg_size)).show()

pos = [269, 75, 34, 64]
#pos = [117, 68, 122, 125]
pos_r = [pos[0], pos[1], pos[0] + pos[2], pos[1] + pos[3]]
roi_pos, roi_rect = calc_roi(img.size, pos)
roi_img = img.crop(box=roi_rect).resize((vgg_size, vgg_size))
pos_in_roi = [pos[0] - roi_pos[0], pos[1] - roi_pos[1], pos[2], pos[3]]
xfac = 1.0 * vgg_size / roi_pos[2]
yfac = 1.0 * vgg_size / roi_pos[3]
pos_in_roi_img = [pos_in_roi[0] * xfac, pos_in_roi[1]
                  * yfac, pos_in_roi[2] * xfac, pos_in_roi[3] * yfac]
feat_pos = [int(n / 8.0) for n in pos_in_roi_img]
print(roi_pos, roi_rect, pos_in_roi, xfac, yfac, pos_in_roi_img, feat_pos)

draw = ImageDraw.Draw(img)
draw.rectangle(pos_r, None, "red")
draw.rectangle(roi_rect, None, "blue")
img.show()

rrr = [int(pos_in_roi_img[0]), int(pos_in_roi_img[1]), int(
    pos_in_roi_img[0] + pos_in_roi_img[2]), int(pos_in_roi_img[1] + pos_in_roi_img[3])]
roi_disp_image = roi_img.copy()
draw = ImageDraw.Draw(roi_disp_image)
draw.rectangle(rrr, None, "red")
roi_disp_image.show()


data = np.array(roi_img.getdata()).reshape(
    (1, vgg_size, vgg_size, 3))
print(data.shape)

sfit = CnnSelNet(name='sfit')
gfit = CnnSelNet(name='gfit')

mask = gauss.gen_gauss_mask(
    (46, 46), feat_pos, sigf=0.8).reshape((1, 46, 46, 1)).T

print("mask.shape", mask.shape)
# exit()


# loss_function =

# cross_entropy = tf.reduce_mean(-tf.reduce_sum(mask *
#                                              tf.log(sfit_conv), reduction_indices=[1]))
#train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# correct_prediction = tf.equal(tf.argmax(sfit_conv, 1), tf.argmax(y_, 1))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
"""
cost = tf.reduce_mean(tf.square(sfit_conv - mask))
#cost = tf.add(tf.abs(sfit_conv - mask))
#optimizer = tf.train.GradientDescentOptimizer(0.001)
optimizer = tf.train.AdamOptimizer(0.01)
sfit_train = optimizer.minimize(cost)
"""


def do_plot(sfit_out, gfit_out):
    f = plt.figure()
    f.add_subplot(1, 2, 1)
    prep = np.swapaxes(np.swapaxes(sfit_out[0], 0, 2), 1, 2)[0]
    img = plt.imshow(prep, cmap='hot', interpolation='nearest')
    f.add_subplot(1, 2, 2)
    prep = np.swapaxes(np.swapaxes(gfit_out[0], 0, 2), 1, 2)[0]
    img = plt.imshow(prep, cmap='hot', interpolation='nearest')
    plt.show()


with tf.Session() as sess:
    # with tf.device("/cpu:0"):
    if True:
        sess.run(tf.initialize_all_variables())
        # setup vgg16
        vgg = vgg16.Vgg16(
            input_size=vgg_size,
            vgg16_npy_path='/informatik2/students/home/3springs/git/tensorflow-vgg/vgg16.npy',
        )
        roi_data = tf.placeholder(
            tf.float32, (1, vgg_size, vgg_size, 3), 'roi_data')
        vgg.build(roi_data)
        # get features from vgg16
        c4, c5 = sess.run(
            [vgg.conv4_3, vgg.conv5_3], feed_dict={roi_data: data})
        # c5 features must be scaled up:
        c5 = c5.repeat(2, axis=1).repeat(2, axis=2)
        print("Feature shapes:", c4.shape, c5.shape)

        # print("MEAN:", np.mean(c5up))
        # print(tf.is_variable_initialized(sfit_conv_bias))
        feed_features = feed_dict = {
            sfit.input: c4, gfit.input: c5, sfit.truth_input: mask, gfit.truth_input: mask}

        epochs = 10000
        show_epochs = 100
        print(
            "Train {} epochs, show masks every {}".format(epochs, show_epochs))
        for epoch in range(epochs + 1):
            print("  {}/{}".format(epoch, epochs))

            # calc cost:
            sfit_cost, gfit_cost = sess.run(
                [sfit.cost, gfit.cost], feed_dict=feed_features)
            print("    COSTs:", sfit_cost, gfit_cost)

            # run features through nets:
            if epoch % show_epochs == 0:
                sfit_out, gfit_out = sess.run(
                    [sfit.conv, gfit.conv], feed_dict=feed_features)
                do_plot(sfit_out, gfit_out)

            if epoch < epochs:
                # train:
                sess.run([sfit.train, gfit.train], feed_dict=feed_features)

        print("Done training")

exit()
data5 = np.swapaxes(np.swapaxes(c5[0], 0, 2), 1, 2)
#data5 = np.fliplr(c4[0])
print(data5.shape)

for i in range(512):
    img = plt.imshow(data5[0 + i], cmap='hot', interpolation='nearest')
    plt.show()

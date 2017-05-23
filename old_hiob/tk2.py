"""
Created on 2016-11-03

@author: Peer Springstübe
"""

import logging
import tensorflow as tf
import tkinter as tk
import numpy as np
from PIL import Image, ImageTk, ImageDraw
from hiob import data_set, vgg16
from hiob.roi import SimpleRoiCalculator
from hiob.rect import Rect
from hiob.gauss import gen_gauss_mask
import matplotlib as plt
from matplotlib import cm

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.DEBUG, format='[%(asctime)s|%(name)s|%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def weight_variable(shape, name=None, stddev=0.1):
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name=None, value=None):
    # initial = tf.constant(, shape=shape)
    if value:
        initial = tf.fill(shape, value)
    else:
        initial = tf.zeros(shape)
    return tf.Variable(initial, name)


class SelectNet(object):

    def __init__(self, session, name, size, feature_count):
        self.session = session
        self.name = name
        self.size = size
        self.feature_count = feature_count
        self.dtype = tf.float32
        with tf.name_scope(self.name):
            self.build_net()
            self.build_trainer()

    def build_net(self):
        self.input_shape = (1, self.size[0], self.size[1], self.feature_count)

        initial = tf.zeros(self.input_shape)
        self.input_variable = tf.Variable(initial)

        self.input = tf.placeholder(
            self.dtype, self.input_shape, name="input")

        self.assign_op = self.input_variable.assign(self.input)
        # dropout layer
        self.dropout = tf.nn.dropout(
            self.input_variable, 0.7, name="dropout")
        # conv1 3x3 kernel, 1 feature
        self.conv1_weight = weight_variable(
            (3, 3, self.feature_count, 1), "conv1_weight", 1e-7)  # stddev1e-7)
        self.conv1_bias = bias_variable((1,), name="conv1_bias", value=0)
        self.conv1 = tf.nn.conv2d(self.dropout, self.conv1_weight, strides=[
                                  1, 1, 1, 1], padding="SAME", name="conv1") + self.conv1_bias

    def build_trainer(self):
        # input for gt:
        self.truth_input_shape = (
            1, self.size[0], self.size[1], 1)
        self.truth_input = tf.placeholder(
            self.dtype, self.truth_input_shape, "truth_input")
        # cost function
        self.learning_rate = 0.000001  # 0.00001
        self.coster = tf.reduce_mean(tf.square(self.conv1 - self.truth_input))
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.trainer = self.optimizer.minimize(self.coster)

        self.grader = self.optimizer.compute_gradients(
            self.coster, [self.input_variable])
        self.grader_placeholder = [
            (tf.placeholder(tf.float32, shape=gr[1].get_shape()), gr[1]) for gr in self.grader]

    def train(self, input_data, truth_data):
        feed = {
            self.input: input_data,
            self.truth_input: truth_data,
        }
        self.session.run([self.assign_op], feed_dict=feed)
        return self.session.run([self.trainer], feed_dict=feed,)

    def cost(self, input_data, truth_data):
        feed = {
            self.input: input_data,
            self.truth_input: truth_data,
        }
        self.session.run([self.assign_op], feed_dict=feed)
        return self.session.run([self.coster], feed_dict=feed,)[0]

    def grade(self, input_data, truth_data):
        feed = {
            self.input: input_data,
            self.truth_input: truth_data,
        }
        # print("FEED:", feed)
        # print(self.grader)
        self.session.run([self.assign_op], feed_dict=feed)
        vals = self.session.run([gr for gr, _ in self.grader], feed_dict=feed)
        #print("VALS", np.shape(vals))
        return vals
        # return self.session.run([self.grader], feed_dict=feed,)[0]

    def forward(self, input_data):
        feed = {
            self.input: input_data,
        }
        self.session.run([self.assign_op], feed_dict=feed)
        return self.session.run([self.conv1], feed_dict=feed,)[0]


class GNet(object):

    def __init__(self, session, name, size, feature_count):
        self.session = session
        self.name = name
        self.size = size
        self.feature_count = feature_count
        self.dtype = tf.float32
        with tf.name_scope(self.name):
            self.build_net()
            self.build_trainer()

    def build_net(self):
        self.input_shape = (1, self.size[0], self.size[1], self.feature_count)
        self.input = tf.placeholder(
            self.dtype, self.input_shape, name="input")
        # conv1 9x9 kernel, 32 features
        self.conv1_weight = weight_variable(
            (9, 9, self.feature_count, 32), "conv1_weight", 1e-14)  # stddev1e-7)
        self.conv1_bias = bias_variable((32,), name="conv1_bias", value=0.1)
        self.conv1 = tf.nn.conv2d(self.input, self.conv1_weight, strides=[
                                  1, 1, 1, 1], padding="SAME", name="conv1") + self.conv1_bias
        # conv2 5x5 kernel, 1 feature
        self.conv2_weight = weight_variable(
            (5, 5, 32, 1), "conv2_weight", 1e-14)  # stddev=1e-7)
        self.conv2_bias = bias_variable((1,), name="conv2_bias", value=0)
        self.conv2 = tf.nn.conv2d(self.conv1, self.conv2_weight, strides=[
                                  1, 1, 1, 1], padding="SAME", name="conv2") + self.conv2_bias

    def build_trainer(self):
        # input for gt:
        self.truth_input_shape = (
            1, self.size[0], self.size[1], 1)
        self.truth_input = tf.placeholder(
            self.dtype, self.truth_input_shape, "truth_input")
        # cost function
        self.learning_rate = 0.00001  # 0.00001
        self.coster = tf.reduce_mean(tf.square(self.conv2 - self.truth_input))
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.trainer = self.optimizer.minimize(self.coster)

    def build_trainer__(self):
        # input for gt:
        self.truth_input_shape = (
            1, self.size[0], self.size[1], 1)
        self.truth_input = tf.placeholder(
            self.dtype, self.truth_input_shape, "truth_input")
        # cost function
        self.learning_rate = 8e-7
        #self.coster = tf.reduce_mean(tf.square(self.conv2 - self.truth_input))
        self.coster = tf.reduce_mean(
            tf.squared_difference(self.conv2, self.truth_input))
        #self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        self.trainer = self.optimizer.minimize(self.coster)

    def build_trainer_g(self):
        # input for gt:
        self.truth_input_shape = (
            1, self.size[0], self.size[1], 1)
        self.truth_input = tf.placeholder(
            self.dtype, self.truth_input_shape, "truth_input")
        # cost function
        self.learning_rate = 0.01
        self.coster = tf.reduce_mean(tf.square(self.conv2 - self.truth_input))
        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        self.trainer = self.optimizer.minimize(self.coster)

    def train(self, input_data, truth_data):
        feed = {
            self.input: input_data,
            self.truth_input: truth_data,
        }
        return self.session.run([self.trainer], feed_dict=feed,)

    def cost(self, input_data, truth_data):
        feed = {
            self.input: input_data,
            self.truth_input: truth_data,
        }
        return self.session.run([self.coster], feed_dict=feed,)[0]

    def forward(self, input_data):
        feed = {
            self.input: input_data,
        }
        return self.session.run([self.conv2], feed_dict=feed,)[0]


class Tracker(object):

    def __init__(self, session, feature_count, sroi_size):
        self.session = session
        self.feature_count = feature_count
        self.sroi_size = sroi_size
        self._build_calc()

    def _build_calc(self):
        logger.info("preparing vgg16")
        self.vgg = vgg16.Vgg16(
            input_size=self.sroi_size[0],
            vgg16_npy_path='/informatik2/students/home/3springs/git/tensorflow-vgg/vgg16.npy',
        )
        self.vgg_input_placeholder = tf.placeholder(
            tf.float32, (1, 368, 368, 3), 'vgg_input_placeholder')
        self.vgg.build(self.vgg_input_placeholder)

        logger.info("creating gnet")
        self.gnet = GNet(
            session=self.session,
            name="gnet", size=(46, 46), feature_count=self.feature_count)
        logger.info("creating snet")
        self.snet = GNet(
            session=self.session,
            name="snet", size=(46, 46), feature_count=self.feature_count)

        logger.info("creating select gnet")
        self.select_gnet = SelectNet(
            session=self.session,
            name="select_gnet", size=(46, 46), feature_count=512)

        logger.info("creating select snet")
        self.select_snet = SelectNet(
            session=self.session,
            name="select_snet", size=(46, 46), feature_count=512)

        # colourmap function
        self.cmap = cm.get_cmap("hot")

    def _reduce(self, data, order, num):
        sh = data.shape
        r = np.zeros((1, sh[1], sh[2], num), dtype=data.dtype)
        for n in range(num):
            # order[n]-1 for values from fcnt
            r[0, :, :, n] = data[0, :, :, order[n]]
        return r

    def calculate_features(self, image):
        feature_input = np.array(image.getdata(), dtype=np.float32).reshape(
            (1, 368, 368, 3))

        c4, c5 = self.session.run(
            [self.vgg.conv4_3, self.vgg.conv5_3], feed_dict={self.vgg_input_placeholder: feature_input})
        c5 = c5.repeat(2, axis=1).repeat(2, axis=2)
        print(c4.shape, c5.shape)
        self.c4 = c4
        self.c5 = c5
        return c4, c5

    def reduce_features(self):
        self.c4_r = self._reduce(
            self.c4, self.snet_ratings, self.feature_count)
        self.c5_r = self._reduce(
            self.c5, self.gnet_ratings, self.feature_count)
        return self.c4, self.c5

    def calculate_target_mask(self, sroi_pos):
        print(sroi_pos)
        f = 46 / 368
#        small_pos = Rect(
# sroi_pos.top * f, sroi_pos.left * f, sroi_pos.width * f, sroi_pos.height
# * f)
        small_pos = Rect(
            sroi_pos.left * f, sroi_pos.top * f, sroi_pos.width * f, sroi_pos.height * f)
        pp = [int(x) for x in small_pos.tuple]
        print(pp)
#        self.tm = gen_gauss_mask(
#            (46, 46), pp, 0.8).T
        self.target_mask = gen_gauss_mask(
            (46, 46), pp, 0.6).T.reshape((1, 46, 46, 1))
#        self.target_mask = gen_gauss_mask(
#            (46, 46), pp, 0.8).reshape((1, 46, 46, 1))
        #mask = self.fg_mask[0].reshape((46, 46))

    def get_target_image(self):
        mask = self.target_mask[0].reshape((46, 46))
        #mask = self.tm
        return Image.fromarray(self.cmap(mask, bytes=True))

    def calculate_net_outputs(self):
        self.snet_output = self.snet.forward(input_data=self.c4_r)
        self.gnet_output = self.gnet.forward(input_data=self.c5_r)

    def get_snet_image(self):
        d = self.snet_output.reshape((46, 46))
        return Image.fromarray(self.cmap(d, bytes=True))

    def get_gnet_image(self):
        d = self.gnet_output.reshape((46, 46))
        return Image.fromarray(self.cmap(d, bytes=True))

    def train_snet(self):
        self.snet.train(self.c4_r, self.target_mask)

    def train_gnet(self):
        self.gnet.train(self.c5_r, self.target_mask)

    def get_gfeature_image(self, pos=None):
        im_num = min(32, self.c5_r.shape[3])
        im = Image.new('RGB', (47 * im_num - 1, 46), color='green')
        for i in range(im_num):
            a = self.c5_r[0, :, :, i]
            a_im = Image.fromarray(self.cmap(a, bytes=True))
            if pos:
                draw = ImageDraw.Draw(a_im)
                draw.rectangle(pos.outer, None, (255, 128, 0, 64))
            im.paste(a_im, (47 * i, 0))
        return im

    def get_sfeature_image(self, pos=None):
        im_num = min(32, self.c4_r.shape[3])
        im = Image.new('RGB', (47 * im_num - 1, 46), color='green')
        for i in range(im_num):
            a = self.c4_r[0, :, :, i]
            a_im = Image.fromarray(self.cmap(a, bytes=True))
            if pos:
                draw = ImageDraw.Draw(a_im)
                draw.rectangle(pos.outer, None, (255, 128, 0, 64))
            im.paste(a_im, (47 * i, 0))
        return im

    # === select ===
    def train_select_snet(self):
        self.select_snet.train(self.c4, self.target_mask)

    def train_select_gnet(self):
        self.select_gnet.train(self.c5, self.target_mask)

    def calculate_select_snet_output(self):
        self.select_snet_output = self.select_snet.forward(input_data=self.c4)

    def calculate_select_gnet_output(self):
        self.select_gnet_output = self.select_gnet.forward(input_data=self.c5)

    def get_select_snet_image(self):
        d = self.select_snet_output.reshape((46, 46))
        return Image.fromarray(self.cmap(d, bytes=True))

    def get_select_gnet_image(self):
        d = self.select_gnet_output.reshape((46, 46))
        return Image.fromarray(self.cmap(d, bytes=True))

    def grade_select_gnet(self):
        diff1 = self.select_gnet_output - self.target_mask
        in_diff1 = self.select_gnet.grade(self.c5, diff1)[0][0]
        diff2 = np.ones_like(self.target_mask)
        in_diff2 = self.select_gnet.grade(self.c5, diff2)[0][0]
        in_data = self.c5[0]
        print("Diff-Shapes:", np.shape(in_diff1),
              np.shape(in_diff2), np.shape(in_data))
        s1 = in_diff1 * in_data
        s2 = 0.5 * in_diff2 * in_data * in_data
        print(np.shape(s1), np.shape(s2))
        sal = s1 + s2
        print(np.shape(sal))
        ss = np.sum(sal, (0, 1))
        print(ss)
        so = np.argsort(ss)
        print(so)
        return so

    def grade_select_snet(self):
        diff1 = self.select_snet_output - self.target_mask
        in_diff1 = self.select_snet.grade(self.c4, diff1)[0][0]
        diff2 = np.ones_like(self.target_mask)
        in_diff2 = self.select_snet.grade(self.c4, diff2)[0][0]
        in_data = self.c4[0]
        print("Diff-Shapes:", np.shape(in_diff1),
              np.shape(in_diff2), np.shape(in_data))
        s1 = in_diff1 * in_data
        s2 = 0.5 * in_diff2 * in_data * in_data
        print(np.shape(s1), np.shape(s2))
        sal = s1 + s2
        print(np.shape(sal))
        ss = np.sum(sal, (0, 1))
        print(ss)
        so = np.argsort(ss)
        print(so)
        return so


class App(tk.Frame):
    sroi_size = (368, 368)
    feature_count = 384
    train_cycles = 25
    select_cycles = 25

    def __init__(self, master=None, session=None):
        super().__init__(master)
        self.session = session
        self.tracked_position = None

        self.pack()

        self.ds = data_set.load_tb100_sample("MotorRolling")

        self.roi_calculator = SimpleRoiCalculator()
        self.create_calc()
        self.create_widgets()

        # initial
        first_capture_image = self.ds.images[0]
        first_pos = self.ds.ground_truth[0]
        first_roi = self.roi_calculator.calculate_roi(
            first_capture_image, first_pos)
        first_sroi_image = first_capture_image.crop(
            first_roi.outer).copy().resize(self.sroi_size)

        # calculate mask for first frame
        spos = self.capture_to_sroi(first_pos, first_roi, self.sroi_size)
        self.tracker.calculate_target_mask(spos)

        logger.info("Getting features of first frame")
        self.tracker.calculate_features(first_sroi_image)

        self.load_frame(0)
        self.select_max = self.select_cycles
        self.select_current = 0
        self.after(10, self.next_select)

        return
        logger.info("reducing features of first frame")
        self.tracker.reduce_features()

        self.tracker.calculate_net_outputs()

        self.load_frame(0)
        self.draw_images()

        self.train_max = self.train_cycles
        self.train_current = 0

        self.after(10, self.next_train)

    def next_select(self):
        self.select_current += 1
        logger.info("Select cycle %d/%d", self.select_current, self.select_max)
        self.tracker.train_select_gnet()
        self.tracker.train_select_snet()
        self.tracker.calculate_select_snet_output()
        self.tracker.calculate_select_gnet_output()
        self.draw_images_select()
        if self.select_current >= self.select_max:
            self.gnet_ratings = self.tracker.grade_select_gnet()
            self.tracker.gnet_ratings = self.gnet_ratings
            self.snet_ratings = self.tracker.grade_select_snet()
            self.tracker.snet_ratings = self.gnet_ratings
            # print(a)

            # to train
            logger.info("reducing features of first frame")
            self.tracker.reduce_features()

            self.tracker.calculate_net_outputs()

            self.load_frame(0)
            self.draw_images()

            self.train_max = self.train_cycles
            self.train_current = 0
            self.after(3, self.next_train)
        else:
            self.after(1, self.next_select)

    def draw_images_select(self):
        cap = self.capture_image.copy()
        draw = ImageDraw.Draw(cap)

        cap_pos = self.tracked_position
        sroi_pos = None
        small_pos = None
        if cap_pos is None:
            cap_pos = self.last_position

        if cap_pos is not None:
            sroi_pos = self.capture_to_sroi(cap_pos, self.roi, self.sroi_size)
            f = 46 / 368
            float_vals = sroi_pos.left * f, sroi_pos.top * \
                f, sroi_pos.width * f, sroi_pos.height * f
            small_pos = Rect([int(x) for x in float_vals])

        if cap_pos is not None:
            draw.rectangle(
                cap_pos.outer, None, (255, 255, 255, 255))
        draw.rectangle(self.roi.outer, None, (0, 255, 255, 255))
        self.capture_label_image = ImageTk.PhotoImage(cap)
        self.capture_label['image'] = self.capture_label_image

        sroi = self.sroi_image.copy()
        draw = ImageDraw.Draw(sroi)
        if sroi_pos is not None:
            draw.rectangle(sroi_pos.outer, None, (255, 0, 255, 255))
        self.sroi_label_image = ImageTk.PhotoImage(sroi)
        self.sroi_label['image'] = self.sroi_label_image

        self.mask_label_image = ImageTk.PhotoImage(
            self.tracker.get_target_image())
        self.mask_label['image'] = self.mask_label_image

        im = self.tracker.get_select_gnet_image()
        if small_pos is not None:
            draw = ImageDraw.Draw(im)
            draw.rectangle(small_pos.outer, None, (0, 0, 255, 128))
        self.gnet_output_label_image = ImageTk.PhotoImage(im)
        self.gnet_output_label['image'] = self.gnet_output_label_image

        im = self.tracker.get_select_snet_image()
        if small_pos is not None:
            draw = ImageDraw.Draw(im)
            draw.rectangle(small_pos.outer, None, (0, 0, 255, 128))
        self.snet_output_label_image = ImageTk.PhotoImage(im)
        self.snet_output_label['image'] = self.snet_output_label_image

    def next_train(self):
        self.train_current += 1

        logger.info("Training, %d/%d", self.train_current, self.train_max)
        self.tracker.train_snet()
        self.tracker.train_gnet()
        self.tracker.calculate_net_outputs()
        cost = self.tracker.gnet.cost(
            input_data=self.tracker.c5_r, truth_data=self.tracker.target_mask)
        logger.info("Cost: %f", cost)
        self.draw_images()

        if self.train_current >= self.train_max:
            self.after(1, self.next_frame)
        else:
            self.after(1, self.next_train)

    def create_dummy_widgets(self):
        self.hi_there = tk.Button(self)
        self.hi_there['text'] = "Hallo World\n(click me)"
        self.hi_there["command"] = self.say_hi
        self.hi_there.pack(side="top")
        self.quit = tk.Button(
            self, text="QUIT", fg="red", command=root.destroy)
        self.quit.pack(side="bottom")

    def create_widgets(self):
        self.current = 0
        self.last_position = self.ds.ground_truth[0]
        self.max = len(self.ds.images)
        self.delay = 1000 // 30
        self.delay = 1

        self.caption_label = tk.Label(self, justify=tk.LEFT)
        self.caption_label.pack()

        self.capture_image = None
        self.capture_label = tk.Label(self)
        self.capture_label.pack()
        self.capture_label_image = None

        self.sroi_image = None
        self.sroi_label = tk.Label(self)
        self.sroi_label.pack()
        self.sroi_label_image = None

        self.mask_label = tk.Label(self)
        self.mask_label.pack()

        self.snet_label = tk.Label(self)
        self.snet_label.pack()
        self.snet_label_image = None

        self.gnet_label = tk.Label(self)
        self.gnet_label.pack()
        self.gnet_label_image = None

        self.snet_output_label = tk.Label(self)
        self.snet_output_label.pack()
        self.snet_output_label_image = None

        self.gnet_output_label = tk.Label(self)
        self.gnet_output_label.pack()
        self.gnet_output_label_image = None

    def create_calc(self):
        self.tracker = Tracker(
            self.session, self.feature_count, sroi_size=self.sroi_size)
        #self.snet = GNet(name="snet", input_features=self.features)
        #self.gnet = GNet(name="gnet", input_features=self.features)
        self.session.run(tf.initialize_all_variables())

    def capture_to_sroi(self, pos, roi, sroi_size):
        """
        Convert rect in capture to rect in scaled roi.
        """
        rx, ry, rw, rh = roi.tuple
        px, py, pw, ph = pos.tuple
        scale_w = sroi_size[0] / rw
        scale_h = sroi_size[1] / rh
        ix = round((px - rx) * scale_w)
        iy = round((py - ry) * scale_h)
        iw = scale_w * pw
        ih = scale_h * ph
        return Rect(ix, iy, iw, ih)

    def sroi_to_capture(self, pos, roi, sroi_size):
        """
        Convert rect in scaled roi to rect in capture.
        """
        rx, ry, rw, rh = roi.tuple
        sx, sy, sw, sh = pos.tuple
        scale_w = sroi_size[0] / rw
        scale_h = sroi_size[1] / rh
        cx = round(sx / scale_w + rx)
        cy = round(sy / scale_h + ry)
        cw = sw / scale_w
        ch = sh / scale_h
        return Rect(cx, cy, cw, ch)

    def draw_images(self):
        cap = self.capture_image.copy()
        draw = ImageDraw.Draw(cap)

        cap_pos = self.tracked_position
        sroi_pos = None
        small_pos = None
        if cap_pos is None:
            cap_pos = self.last_position

        if cap_pos is not None:
            sroi_pos = self.capture_to_sroi(cap_pos, self.roi, self.sroi_size)
            f = 46 / 368
            float_vals = sroi_pos.left * f, sroi_pos.top * \
                f, sroi_pos.width * f, sroi_pos.height * f
            small_pos = Rect([int(x) for x in float_vals])

        if cap_pos is not None:
            draw.rectangle(
                cap_pos.outer, None, (255, 255, 255, 255))
        draw.rectangle(self.roi.outer, None, (0, 255, 255, 255))
        self.capture_label_image = ImageTk.PhotoImage(cap)
        self.capture_label['image'] = self.capture_label_image

        sroi = self.sroi_image.copy()
        draw = ImageDraw.Draw(sroi)
        if sroi_pos is not None:
            draw.rectangle(sroi_pos.outer, None, (255, 0, 255, 255))
        self.sroi_label_image = ImageTk.PhotoImage(sroi)
        self.sroi_label['image'] = self.sroi_label_image

        self.mask_label_image = ImageTk.PhotoImage(
            self.tracker.get_target_image())
        self.mask_label['image'] = self.mask_label_image

        self.gnet_label_image = ImageTk.PhotoImage(
            self.tracker.get_gfeature_image(small_pos))
        self.gnet_label['image'] = self.gnet_label_image

        # self.snet_label_image = ImageTk.PhotoImage(
        #    self.tracker.get_snet_image())
        self.snet_label_image = ImageTk.PhotoImage(
            self.tracker.get_sfeature_image(small_pos))
        self.snet_label['image'] = self.snet_label_image

        im = self.tracker.get_gnet_image()
        if small_pos is not None:
            draw = ImageDraw.Draw(im)
            draw.rectangle(small_pos.outer, None, (0, 0, 255, 128))
        self.gnet_output_label_image = ImageTk.PhotoImage(im)
        self.gnet_output_label['image'] = self.gnet_output_label_image

        im = self.tracker.get_snet_image()
        if small_pos is not None:
            draw = ImageDraw.Draw(im)
            draw.rectangle(small_pos.outer, None, (0, 0, 255, 128))
        self.snet_output_label_image = ImageTk.PhotoImage(im)
        self.snet_output_label['image'] = self.snet_output_label_image

    def load_frame(self, n):
        self.capture_image = self.ds.images[n]
        # calculate roi
        self.roi = self.roi_calculator.calculate_roi(
            self.capture_image, self.last_position)
        self.sroi_image = self.capture_image.crop(
            self.roi.outer).copy().resize(self.sroi_size)

    def next_frame(self):
        # go to next frame:
        self.current += 1
        if self.current >= self.max:
            return

        cap_text = "Sample: {} / {} - Frame: {}/{}".format(
            self.ds.set_name, self.ds.sample_name, self.current + 1, len(self.ds.images))
        self.caption_label['text'] = cap_text

        self.load_frame(self.current)

        spos = self.capture_to_sroi(
            self.last_position, self.roi, self.sroi_size)
        # spos = self.capture_to_sroi(
        #    self.ds.ground_truth[self.current], self.roi, self.sroi_size)
        self.tracker.calculate_target_mask(spos)

        self.tracker.calculate_features(self.sroi_image)
        self.tracker.reduce_features()
        self.tracker.calculate_net_outputs()
        #c4, c5 = self.get_features(self.sroi_image)
        #print(c4.shape, c5.shape)

        #c4, c5 = self.reduce_features(c4, c5)
        #print(c4.shape, c5.shape)

        # cheat and get gt as last localisation
        self.tracked_position = self.ds.ground_truth[self.current]

        spos = self.capture_to_sroi(
            self.tracked_position, self.roi, self.sroi_size)
        self.tracker.calculate_target_mask(spos)

        #
        self.draw_images()

        # remember current position for next frame
        self.last_position = self.tracked_position

        self.after(self.delay, self.next_frame)

with tf.Session() as sess:
    root = tk.Tk()
    app = App(master=root, session=sess)
    app.mainloop()

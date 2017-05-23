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
from matplotlib import cm
import scipy.ndimage

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
    DROPOUT_KEEP = 0.7
    LEARNING_RATE = 0.000001

    def __init__(self, session, name=None, input_shape=None, dtype=tf.float32, dropout_keep=None, learning_rate=None):
        self.session = session
        self.name = name
        self.input_shape = input_shape  # (1, 46, 46, 512)
        self.target_shape = input_shape[:3] + (1,)  # (1, 46, 46, 1)
        self.dtype = dtype
        if dropout_keep is None:
            self.dropout_keep = self.DROPOUT_KEEP
        else:
            self.dropout_keep = dropout_keep
        if learning_rate is None:
            self.learning_rate = self.LEARNING_RATE
        else:
            self.learning_rate = learning_rate
        logger.info("creating SelectNet for input shape %s", self.input_shape)
        with tf.name_scope(self.name):
            self._build_variables()
            self._build_net()
            self._build_calculation()

    def _build_variables(self):
        # create variable for input values:
        input_initial = tf.zeros(
            self.input_shape,
            name="input_initial",
            dtype=self.dtype,
        )
        self.input_variable = tf.Variable(
            initial_value=input_initial,
            name="input_variable",
            dtype=self.dtype,
            trainable=False,
        )
        self.input_placeholder = tf.placeholder(
            dtype=self.dtype,
            shape=self.input_shape,
            name="input_placeholder",
        )
        self.input_assign = self.input_variable.assign(self.input_placeholder)

        # create variable for target values:
        target_initial = tf.zeros(
            self.target_shape,
            name="target_initial",
            dtype=self.dtype,
        )
        self.target_variable = tf.Variable(
            initial_value=target_initial,
            name="target_variable",
            dtype=self.dtype,
            trainable=False,
        )
        self.target_placeholder = tf.placeholder(
            dtype=self.dtype,
            shape=self.target_shape,
            name="target_placeholder",
        )
        self.target_assign = self.target_variable.assign(
            self.target_placeholder)

    def _build_net(self):
        # dropout layer:
        self.dropout = tf.nn.dropout(
            self.input_variable,
            keep_prob=self.dropout_keep,
            name="dropout",
        )
        # conv1 3x3 kernel
        self.conv1_weight = weight_variable(
            (3, 3, self.input_shape[3], 1), "conv1_weight", 1e-7)  # stddev1e-7)
        self.conv1_bias = bias_variable((1,), name="conv1_bias", value=0)
        self.conv1 = tf.nn.conv2d(self.dropout, self.conv1_weight, strides=[
                                  1, 1, 1, 1], padding="SAME", name="conv1") + self.conv1_bias

    def _build_calculation(self):
        # cost function
        self.cost_function = tf.reduce_mean(
            tf.square(self.conv1 - self.target_variable))
        self.optimizer = tf.train.AdamOptimizer(
            self.learning_rate, name="optimizer")
        self.trainer = self.optimizer.minimize(
            self.cost_function, name="trainer")
        self.gradient_function = self.optimizer.compute_gradients(
            self.cost_function, [self.input_variable])

    def set_input(self, input_data):
        self.session.run(
            [self.input_assign],
            feed_dict={self.input_placeholder: input_data},
        )

    def set_target(self, target_data):
        self.session.run(
            [self.target_assign],
            feed_dict={self.target_placeholder: target_data},
        )

    def train(self):
        self.session.run([self.trainer])

    def cost(self):
        return self.session.run([self.cost_function])[0]

    def forward(self):
        return self.session.run([self.conv1])[0]

    def gradient(self):
        vals = self.session.run([gr for gr, _ in self.gradient_function])
        return vals


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
            (9, 9, self.feature_count, 32), "conv1_weight", 1e-7)  # stddev1e-7)
        self.conv1_bias = bias_variable((32,), name="conv1_bias", value=0.1)
        self.conv1 = tf.nn.conv2d(self.input, self.conv1_weight, strides=[
                                  1, 1, 1, 1], padding="SAME", name="conv1") + self.conv1_bias
        # conv2 5x5 kernel, 1 feature
        self.conv2_weight = weight_variable(
            (5, 5, 32, 1), "conv2_weight", 1e-7)  # stddev=1e-7)
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


class Frame(object):

    def __init__(self, capture_image=None, previous_position=None, ground_truth=None, number=None, sample=None):
        self.number = number
        self.sample = sample
        self.capture_image = capture_image
        self.sroi_image = None

        self.previous_position = previous_position
        self.predicted_position = None
        self.roi = None
        self.ground_truth = ground_truth

        self.features = None
        self.reduced_features = None
        self.prediction_mask = None
        self.image_mask = None

    def __str__(self):
        if self.sample:
            return "<Frame %s/%s#%04d/%04d>" % (self.sample.set_name, self.sample.sample_name, self.number, len(self.sample.images))
        else:
            return "<Frame None#" + self.number + ">"


def loc2affgeo(location, particle_size=64):
    x, y, w, h = location
    cx = x + (w - 1) / 2
    cy = y + (h - 1) / 2
    gw = w / particle_size
    gh = h / w
    geo = [cx, cy, gw, gh]
    return geo


def affgeo2loc(geo, particle_size=64):
    cx, cy, pw, ph = geo
    w = pw * particle_size
    h = ph * w
    x = cx - (w - 1) / 2
    y = cy - (h - 1) / 2
    return [x, y, w, h]


class Tracker(object):

    def __init__(self, session, feature_count, name="HiobTrackerOne"):
        self.session = session
        self.name = name
        self.sroi_size = (368, 368)
        self.mask_size = (46, 46)
        self.mask_scale = (8, 8)

        self.particle_count = 600

        self.data_set = None
        self.initial_frame = None
        self.sequence_length = None
        self.current_frame = None
        self.current_frame_number = None
        self.positions = None

        self.feature_count = feature_count

        # size of a single feature:
        self.feature_size = (1, self.mask_size[0], self.mask_size[1], 1)
        # impact of individual features on target prediction:
        self.feature_ratings = None
        # feature indices ordered by impact:
        self.feature_order = None

        logger.info("creating roi calculator")
        self.roi_calculator = SimpleRoiCalculator()

        # colour map function
        self.cmap = cm.get_cmap("hot")

        with tf.name_scope(self.name):
            self._build_feature_net()
            self._build_selector_net()
            self._build_processor_net()

    def _build_feature_net(self):
        logger.info("creating pretrained vgg16 net as feature extractor")
        self.vgg = vgg16.Vgg16(
            input_size=self.sroi_size[0],
            vgg16_npy_path='/informatik2/students/home/3springs/git/tensorflow-vgg/vgg16.npy',
        )
        self.vgg_input_placeholder = tf.placeholder(
            tf.float32, (1, 368, 368, 3), 'vgg_input_placeholder')
        self.vgg.build(self.vgg_input_placeholder)

    def _build_selector_net(self):
        logger.info("creating select gnet selector net")
        self.gnet_selector = SelectNet(
            session=self.session,
            name="gnet_selector",
            input_shape=(1, 46, 46, 512),
        )
        logger.info("creating select snet selector net")
        self.snet_selector = SelectNet(
            session=self.session,
            name="gnet_selector",
            input_shape=(1, 46, 46, 512),
        )

    def _build_processor_net(self):
        logger.info("creating gnet")
        self.gnet = GNet(
            session=self.session,
            name="gnet", size=(46, 46), feature_count=self.feature_count)
        logger.info("creating snet")
        self.snet = GNet(
            session=self.session,
            name="snet", size=(46, 46), feature_count=self.feature_count)

    def capture_to_sroi(self, pos, roi):
        """
        Convert rect in capture to rect in scaled roi.
        """
        rx, ry, rw, rh = roi.tuple
        px, py, pw, ph = pos.tuple
        scale_w = self.sroi_size[0] / rw
        scale_h = self.sroi_size[1] / rh
        ix = round((px - rx) * scale_w)
        iy = round((py - ry) * scale_h)
        iw = scale_w * pw
        ih = scale_h * ph
        return Rect(ix, iy, iw, ih)

    def sroi_to_capture(self, pos, roi):
        """
        Convert rect in scaled roi to rect in capture.
        """
        rx, ry, rw, rh = roi.tuple
        sx, sy, sw, sh = pos.tuple
        scale_w = self.sroi_size[0] / rw
        scale_h = self.sroi_size[1] / rh
        cx = round(sx / scale_w + rx)
        cy = round(sy / scale_h + ry)
        cw = sw / scale_w
        ch = sh / scale_h
        return Rect(cx, cy, cw, ch)

    def sroi_to_mask(self, sroi_pos):
        return Rect(
            int(sroi_pos.left / self.mask_scale[0]),
            int(sroi_pos.top / self.mask_scale[1]),
            int(sroi_pos.width / self.mask_scale[0]),
            int(sroi_pos.height / self.mask_scale[1]),
        )

    def mask_to_sroi(self, mask_pos):
        return Rect(
            int(mask_pos.left / self.mask_scale[0]),
            int(mask_pos.top / self.mask_scale[1]),
            int(mask_pos.width / self.mask_scale[0]),
            int(mask_pos.height / self.mask_scale[1]),
        )

    def capture_to_mask(self, pos, roi):
        """
        Convert rect in capture to rect in mask roi.
        """
        rx, ry, rw, rh = roi.tuple
        px, py, pw, ph = pos.tuple
        scale_w = (self.sroi_size[0] / rw) / self.mask_scale[0]
        scale_h = (self.sroi_size[1] / rh) / self.mask_scale[0]
        ix = int(round((px - rx) * scale_w))
        iy = int(round((py - ry) * scale_h))
        iw = int(round(scale_w * pw))
        ih = int(round(scale_h * ph))
        return Rect(ix, iy, iw, ih)

    def mask_to_capture(self, pos, roi):
        """
        Convert rect in mask to rect in capture.
        """
        rx, ry, rw, rh = roi.tuple
        sx, sy, sw, sh = pos.tuple
        scale_w = (self.sroi_size[0] / rw) / self.mask_scale[0]
        scale_h = (self.sroi_size[1] / rh) / self.mask_scale[0]
        cx = int(round(sx / scale_w + rx))
        cy = int(round(sy / scale_h + ry))
        cw = int(round(sw / scale_w))
        ch = int(round(sh / scale_h))
        return Rect(cx, cy, cw, ch)

    def load_sample(self, set_name, sample_name):
        if set_name == 'tb100':
            self.sample = data_set.load_tb100_sample(sample_name)
            self.initial_frame = Frame(
                capture_image=self.sample.images[0],
                previous_position=self.sample.ground_truth[0],
                ground_truth=self.sample.ground_truth[0],
                number=0,
                sample=self.sample,
            )
            # first frame has ground truth position given:
            self.initial_frame.predicted_position = self.initial_frame.ground_truth
        else:
            raise ValueError("unknown dataset '%s'" % set_name)

    def calculate_frame_roi(self, frame):
        logger.info("Calculating ROI for frame #%d", frame.number)
        roi = self.roi_calculator.calculate_roi(
            frame.capture_image, frame.previous_position)
        logger.info("ROI for %s: %s", frame, roi)
        frame.roi = roi

    def create_frame_sroi_image(self, frame):
        frame.sroi_image = frame.capture_image.crop(
            frame.roi.outer).copy().resize(self.sroi_size)

    def extract_frame_features(self, frame):
        logger.info("Extracting features for %s", frame)
        # get np-array from image data:
        feature_input = np.array(frame.sroi_image.getdata(), dtype=np.float32).reshape(
            (1, 368, 368, 3))
        # execute vgg16 to get layers conv4 and conv5 output:
        c4, c5 = self.session.run(
            [self.vgg.conv4_3, self.vgg.conv5_3], feed_dict={self.vgg_input_placeholder: feature_input})
        # scale up c5 to fit size of c4:
        c5 = c5.repeat(2, axis=1).repeat(2, axis=2)
        # store features in frame:
        frame.features = {'conv4': c4, 'conv5': c5}

    def create_frame_target_mask(self, frame):
        sroi_pos = self.capture_to_sroi(frame.previous_position, frame.roi)
        mask_pos = self.sroi_to_mask(sroi_pos)
        mask_shape = (1, self.mask_size[0], self.mask_size[1], 1)
        target_mask = gen_gauss_mask(
            self.mask_size, mask_pos, 0.6).T.reshape(mask_shape)
        frame.target_mask = target_mask

    def prepare_frame_feature_selection(self, frame):
        self.current_frame = frame
        self.calculate_frame_roi(frame)
        self.create_frame_sroi_image(frame)
        # create target mask (how output of gnet and snet should look like):
        self.create_frame_target_mask(frame)
        self.extract_frame_features(frame)
        # fill selector nets with data:
        self.snet_selector.set_input(frame.features['conv4'])
        self.gnet_selector.set_input(frame.features['conv5'])
        self.snet_selector.set_target(frame.target_mask)
        self.gnet_selector.set_target(frame.target_mask)

    def train_feature_selection(self):
        self.snet_selector.train()
        self.gnet_selector.train()
        return self.snet_selector.cost(), self.gnet_selector.cost()

    def _evaluate_selection_rating(self, net, in_data, target_mask):
        diff1 = self.snet_selector.forward() - target_mask
        net.set_target(diff1)
        in_diff1 = net.gradient()[0][0]
        diff2 = np.ones_like(target_mask)
        net.set_target(diff2)
        in_diff2 = net.gradient()[0][0]
        sal = (in_diff1 * in_data) + (0.5 * in_diff2 * in_data * in_data)
        print("Diff-Shapes:", np.shape(in_diff1),
              np.shape(in_diff2), np.shape(in_data))
        print(np.shape(sal))
        order = np.sum(sal, (0, 1))
        print(np.shape(order))
        return order

    def evaluate_feature_selection(self):
        logger.info("Evaluating feature impact")
        frame = self.current_frame
        snet_rating = self._evaluate_selection_rating(
            self.snet_selector, frame.features['conv4'][0], frame.target_mask)
        snet_order = np.argsort(snet_rating)
        gnet_rating = self._evaluate_selection_rating(
            self.gnet_selector, frame.features['conv5'][0], frame.target_mask)
        gnet_order = np.argsort(gnet_rating)
        self.feature_ratings = {'conv4': snet_rating, 'conv5': gnet_rating}
        self.feature_order = {'conv4': snet_order, 'conv5': gnet_order}
        logger.info("feature_rating: %s", self.feature_ratings)
        logger.info("feature_order: %s", self.feature_order)

    def _feature_reduce(self, data, order, num):
        sh = data.shape
        r = np.zeros((1, sh[1], sh[2], num), dtype=data.dtype)
        for n in range(num):
            r[0, :, :, n] = data[0, :, :, order[n]]
        return r

    def reduce_frame_features(self, frame):
        reduced = {}
        for name, feature in frame.features.items():
            reduced[name] = self._feature_reduce(
                feature, self.feature_order[name], self.feature_count)
        frame.reduced_features = reduced

    def process_frame_features(self, frame=None):
        if frame is None:
            frame = self.current_frame
        m4 = self.snet.forward(frame.reduced_features['conv4'])
        m5 = self.gnet.forward(frame.reduced_features['conv5'])
        frame.prediction_mask = {'conv4': m4, 'conv5': m5}

    def train_snet(self):
        frame = self.current_frame
        self.snet.train(frame.reduced_features['conv4'], frame.target_mask)

    def train_gnet(self):
        frame = self.current_frame
        self.gnet.train(frame.reduced_features['conv5'], frame.target_mask)

    def load_next_frame(self):
        prev_pos = self.current_frame.predicted_position
        self.current_frame_number += 1
        self.current_frame = Frame(
            capture_image=self.sample.images[self.current_frame_number],
            previous_position=prev_pos,
            ground_truth=self.sample.ground_truth[self.current_frame_number],
            number=self.current_frame_number,
            sample=self.sample,
        )

    # === tracking/distracting ===

    def generate_geo_particles(self, geo, img_size):
        # geo = loc2affgeo(loc)
        geos = np.tile(geo, (self.particle_count, 1)).T
        r = np.random.randn(4, self.particle_count)
        f = np.tile([10, 10, .01, .1], (self.particle_count, 1)).T
        rn = np.multiply(r, f)

        geos += rn
        #
        if False:
            geos[2, geos[2, :] < 0.05] = 0.05
            geos[2, geos[2, :] > 0.95] = 0.95
            geos[3, geos[3, :] < 0.10] = 0.10
            geos[3, geos[3, :] > 10.0] = 10.0

            w = img_size[0]
            h = img_size[1]
            geos[0, geos[0, :] < (0.05 * w)] = 0.05 * w
            geos[0, geos[0, :] > (0.95 * w)] = 0.95 * w
            geos[1, geos[1, :] < (0.05 * h)] = 0.05 * h
            geos[1, geos[1, :] > (0.95 * h)] = 0.95 * h
        #
        return (geos + rn).T

    def generate_particles(self, loc, img_size):
        geo = loc2affgeo(loc)
        geos = self.generate_geo_particles(geo, img_size)
        locs = [affgeo2loc(g) for g in geos]
        return locs

    def upscale_mask(self, mask, roi, image_size):
        relation = roi.width / self.mask_size[0], \
            roi.height / self.mask_size[1]
        roi_mask = scipy.ndimage.zoom(mask.reshape(self.mask_size), relation)
        roi_mask[roi_mask < 0.1] = -.5
        img_mask = np.zeros(image_size)
        img_mask[int(roi.top): int(roi.bottom),
                 int(roi.left): int(roi.right)] = roi_mask
        return img_mask

    def position_quality(self, image_mask, pos):
        #logger.info("QUALI: %s, %s", image_mask.shape, pos)
        if pos.left < 0 or pos.top < 0 or pos.right >= image_mask.shape[1] or pos.bottom >= image_mask.shape[0]:
            return -100
        if pos.width < 16 or pos.height < 16:
            return -100
        inner = (image_mask[
            int(pos.top):int(pos.bottom),
            int(pos.left):int(pos.right)]).sum()
        outer = (image_mask).sum() - inner
        # punish area outside the image:
        # TODO:
        return inner - outer

    def predict_frame_position(self, frame):
        logger.info("Predicting position for frame %s", frame)
        mask = frame.prediction_mask['conv5']
        img_size = [frame.capture_image.size[1], frame.capture_image.size[0]]
        img_mask = self.upscale_mask(mask, frame.roi, img_size)
        frame.image_mask = img_mask
        locs = self.generate_particles(
            frame.previous_position, frame.capture_image.size)
        quals = [self.position_quality(img_mask, Rect(l)) for l in locs]
        #quals = []
        # for n, l in enumerate(locs):
        #    r = Rect(l)
        #    q = self.position_quality(img_mask, r)
        #    logger.info("%d, %r: %f", n, r, q)
        #    quals.append(q)
        best_arg = np.argmax(quals)
        frame.predicted_position = Rect(locs[best_arg])
        logger.info("Prediction: %s", frame.predicted_position)
        return frame.predicted_position

    # === images ===

    def get_snet_selector_image(self):
        d = self.snet_selector.forward().reshape(self.mask_size)
        im = Image.fromarray(self.cmap(d, bytes=True))
        return im

    def get_gnet_selector_image(self):
        d = self.gnet_selector.forward().reshape(self.mask_size)
        im = Image.fromarray(self.cmap(d, bytes=True))
        return im

    def get_reduced_feature_image(self, feature_name, pos=None):
        frame = self.current_frame
        feat = frame.reduced_features[feature_name]
        im_num = min(32, feat.shape[3])

        im = Image.new(
            'RGB', ((self.mask_size[0] + 1) * im_num - 1, self.mask_size[1]), color='green')
        for i in range(im_num):
            a = feat[0, :, :, i]
            a_im = Image.fromarray(self.cmap(a, bytes=True))
            if pos:
                draw = ImageDraw.Draw(a_im)
                draw.rectangle(pos.outer, None, (255, 128, 0, 64))
            im.paste(a_im, ((self.mask_size[0] + 1) * i, 0))
        return im

    def get_snet_image(self, pos=None):
        frame = self.current_frame
        im = Image.fromarray(
            self.cmap(frame.prediction_mask['conv4'].reshape(self.mask_size), bytes=True))
        if pos:
            draw = ImageDraw.Draw(im)
            draw.rectangle(pos.outer, None, (255, 128, 0, 64))
        return im

    def get_gnet_image(self, pos=None):
        frame = self.current_frame
        im = Image.fromarray(
            self.cmap(frame.prediction_mask['conv5'].reshape(self.mask_size), bytes=True))
        if pos:
            draw = ImageDraw.Draw(im)
            draw.rectangle(pos.outer, None, (255, 128, 0, 64))
        return im

    def get_image_mask_image(self, frame=None, pos=None):
        if frame is None:
            frame = self.current_frame
        if frame.image_mask is None:
            return None
        im = Image.fromarray(
            self.cmap(frame.image_mask, bytes=True))
        if pos:
            draw = ImageDraw.Draw(im)
            draw.rectangle(pos.outer, None, (255, 129, 0, 255))
        return im


class App(tk.Frame):
    sroi_size = (368, 368)
    feature_count = 128  # 384
    train_cycles = 50
    select_cycles = 50

    def __init__(self, master=None, session=None):
        super().__init__(master)

        # init gui stuff:
        self.create_widgets()
        self.pack()

        self.session = session

        self.tracker = Tracker(
            session=self.session, feature_count=self.feature_count)
        self.session.run(tf.initialize_all_variables())
        self.tracker.load_sample("tb100", "Singer2")

        self.run_select()

    def create_widgets(self):
        self.caption_label = tk.Label(self, justify=tk.LEFT)
        self.caption_label.pack()

        self.capture_label = tk.Label(self)
        self.capture_label.pack()
        self.capture_label_image = None

        self.image_mask_label = tk.Label(self)
        self.image_mask_label.pack()
        self.image_mask_label_image = None

        self.sroi_label = tk.Label(self)
        self.sroi_label.pack()
        self.sroi_label_image = None

        self.snet_feature_label = tk.Label(self)
        self.snet_feature_label.pack()
        self.snet_feature_label_image = None

        self.gnet_feature_label = tk.Label(self)
        self.gnet_feature_label.pack()
        self.gnet_feature_label_image = None

        self.snet_label = tk.Label(self)
        self.snet_label.pack()
        self.snet_label_image = None

        self.gnet_label = tk.Label(self)
        self.gnet_label.pack()
        self.gnet_label_image = None

    def run_select(self):
        self.select_current = 0
        self.select_max = self.select_cycles

        self.tracker.prepare_frame_feature_selection(
            self.tracker.initial_frame)

        self.caption_label[
            'text'] = "%s feature selection" % self.tracker.current_frame
        self.draw_select()

        self.next_select()

    def draw_select(self):
        im_c = self.tracker.current_frame.capture_image.copy()
        self.capture_label_image = ImageTk.PhotoImage(im_c)
        self.capture_label['image'] = self.capture_label_image

        if self.tracker.current_frame.sroi_image is not None:
            im_sroi = self.tracker.current_frame.sroi_image.copy()
            self.sroi_label_image = ImageTk.PhotoImage(im_sroi)
            self.sroi_label['image'] = self.sroi_label_image

        im_s = self.tracker.get_snet_selector_image()
        self.snet_label_image = ImageTk.PhotoImage(im_s)
        self.snet_label['image'] = self.snet_label_image

        im_g = self.tracker.get_gnet_selector_image()
        self.gnet_label_image = ImageTk.PhotoImage(im_g)
        self.gnet_label['image'] = self.gnet_label_image

    def next_select(self):
        self.select_current += 1
        logger.info(
            "Selection training %d/%d", self.select_current, self.select_max)
        cost = self.tracker.train_feature_selection()
        self.draw_select()
        logger.info("COST: %s", cost)
        if self.select_current < self.select_max:
            self.after(10, self.next_select)
        else:
            logger.info("Feature selection completed")
            self.tracker.evaluate_feature_selection()
            self.run_initial_training()

    def run_initial_training(self):
        self.train_current = 0
        self.train_max = self.train_cycles

        self.tracker.reduce_frame_features(self.tracker.initial_frame)
        self.tracker.process_frame_features()
        self.draw_running()

        self.next_initial_training()

    def next_initial_training(self):
        self.train_current += 1
        logger.info(
            "Initial training %d/%d", self.train_current, self.train_max)

        self.tracker.train_snet()
        self.tracker.train_gnet()
        self.tracker.process_frame_features()
        self.draw_running()

        if self.train_current < self.train_max:
            self.after(1, self.next_initial_training)
        else:
            self.run_tracking()

    def draw_running(self):
        frame = self.tracker.current_frame

        self.caption_label['text'] = frame.__str__()

        capture_pos = frame.predicted_position or frame.previous_position
        roi = frame.roi
        if capture_pos is not None and roi is not None:
            sroi_pos = self.tracker.capture_to_sroi(capture_pos, roi)
            mask_pos = self.tracker.capture_to_mask(capture_pos, roi)
        else:
            sroi_pos = None
            mask_pos = None

        im_c = frame.capture_image.copy()
        draw = ImageDraw.Draw(im_c)
        if capture_pos:
            draw.rectangle(capture_pos.outer, None, (255, 255, 0, 255))
        if roi:
            draw.rectangle(roi.outer, None, (0, 255, 255, 255))
        self.capture_label_image = ImageTk.PhotoImage(im_c)
        self.capture_label['image'] = self.capture_label_image

        im_sroi = frame.sroi_image.copy()
        if sroi_pos:
            draw = ImageDraw.Draw(im_sroi)
            draw.rectangle(sroi_pos.outer, None, (255, 255, 0, 255))
        self.sroi_label_image = ImageTk.PhotoImage(im_sroi)
        self.sroi_label['image'] = self.sroi_label_image

        im_fs = self.tracker.get_reduced_feature_image('conv4', mask_pos)
        self.snet_feature_label_image = ImageTk.PhotoImage(im_fs)
        self.snet_feature_label['image'] = self.snet_feature_label_image

        im_fg = self.tracker.get_reduced_feature_image('conv5', mask_pos)
        self.gnet_feature_label_image = ImageTk.PhotoImage(im_fg)
        self.gnet_feature_label['image'] = self.gnet_feature_label_image

        im_s = self.tracker.get_snet_image(mask_pos)
        self.snet_label_image = ImageTk.PhotoImage(im_s)
        self.snet_label['image'] = self.snet_label_image

        im_g = self.tracker.get_gnet_image(mask_pos)
        self.gnet_label_image = ImageTk.PhotoImage(im_g)
        self.gnet_label['image'] = self.gnet_label_image

        im_im = self.tracker.get_image_mask_image(frame, capture_pos)
        if im_im:
            self.image_mask_label_image = ImageTk.PhotoImage(im_im)
            self.image_mask_label['image'] = self.image_mask_label_image

    def run_tracking(self):
        self.tracker.current_frame_number = 0
        self.next_tracking()

    def next_tracking(self):
        try:
            self.tracker.load_next_frame()
        except IndexError:
            logger.info("Tracking done")
            return
        self.tracker.calculate_frame_roi(self.tracker.current_frame)
        self.tracker.create_frame_sroi_image(self.tracker.current_frame)
        self.tracker.extract_frame_features(self.tracker.current_frame)
        self.tracker.reduce_frame_features(self.tracker.current_frame)
        self.tracker.process_frame_features(self.tracker.current_frame)
        self.tracker.predict_frame_position(self.tracker.current_frame)
        # self.tracker.current_frame.predicted_position =
        # self.tracker.current_frame.ground_truth # TODO: cheating!!!
        self.draw_running()
        self.after(1, self.next_tracking)


with tf.Session() as sess:
    root = tk.Tk()
    app = App(master=root, session=sess)
    app.mainloop()

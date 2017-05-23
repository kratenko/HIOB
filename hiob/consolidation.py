"""
Created on 2016-11-29

@author: Peer Springstübe
"""
import logging
from collections import OrderedDict

import tensorflow as tf
import numpy as np

import hiob.base
from hiob.netbuilder import BuiltNet

logger = logging.getLogger(__name__)


class Consolidator(hiob.base.HiobModule):

    def consolidate_features(self, state, frame):
        raise NotImplementedError()


class SingleNetConsolidator(Consolidator):

    def __init__(self):
        self.dtype = tf.float32

    def configure(self, configuration):
        self.configuration = configuration
        cconf = configuration['consolidator']
        self.max_iterations = cconf['max_iterations']
        self.min_cost = cconf['min_cost']
        self.sigma_train = cconf['sigma_train']
        self.sigma_update = cconf['sigma_update']
        self.update_threshold = cconf['update_threshold']
        self.update_lower_threshold = cconf['update_lower_threshold']
        self.update_frame_store_size = cconf['update_frame_store_size']
        self.update_keep_initial_frame = cconf['update_keep_initial_frame']
        self.update_initial_factor = cconf['update_initial_factor']
        self.update_max_iterations = cconf['update_max_iterations']
        self.update_use_quality = cconf['update_use_quality']
        self.update_current_factor = cconf['update_current_factor']
        self.update_max_frames = cconf['update_max_frames']
        self.update_min_frames = cconf['update_min_frames']
        self.net_configuration = cconf['net']
        # read feature configuration:
        self.feature_counts = OrderedDict()
        for name, num in configuration['features']:
            self.feature_counts[name] = num
        self.total_feature_count = sum(self.feature_counts.values())
        #

    def setup(self, session):
        self.session = session

    def setup_tracking(self, state, output_features):
        # calculate input shape:
        mask_size = self.configuration['mask_size']
        self.input_shape = (
            None, mask_size[0], mask_size[1], self.total_feature_count)
        logger.info("Input shape for ConsolidatorNet: %s", self.input_shape)
        #
        net = BuiltNet(
            self.session,
            self.net_configuration,
            input_shape=self.input_shape,
        )
        state['net'] = net
        # concat features
        placeholders = OrderedDict()
        with tf.name_scope('consolidator'):
            for name, cnt in self.feature_counts.items():
                shape = (None, mask_size[0], mask_size[1], cnt)
                ph = tf.placeholder(
                    dtype=self.dtype,
                    shape=shape,
                    name="concat_placeholder_" + name,
                )
                placeholders[name] = ph
            self.concatenated = tf.concat(3, list(placeholders.values()))
            self.concatenation_placeholders = placeholders
            #
        # frames we keep in memory to use for update of net
        state['stored_frames'] = {}
        # number of last frame we updated consolidator on. Starts on first
        # frame, obviously:
        state['last_update_frame'] = 1

    def _concat_features(self, features):
        feed_dict = {}
        for name, placeholder in self.concatenation_placeholders.items():
            feed_dict[placeholder] = features[name]
        return self.session.run(self.concatenated, feed_dict=feed_dict)

    def _forward(self, state, features):
        con = self._concat_features(features)
        return state['net'].forward(input_data=con)

    def _cost(self, state, features, target):
        con = self._concat_features(features)
        return state['net'].cost(input_data=con, target_data=target)

    def _train(self, state, features, target):
        con = self._concat_features(features)
        return state['net'].train(input_data=con, target_data=target)

    def start_training(self, state, frame):
        assert frame.did_reduction
        state['last_iteration'] = 0

    def training_step(self, state, frame):
        assert frame.did_reduction
        state['last_iteration'] += 1
        self._train(state, frame.features, frame.target_mask)

    def training_cost(self, state, frame):
        assert frame.did_reduction
        return self._cost(state, frame.features, frame.target_mask)

    def training_done(self, state, frame):
        assert frame.did_reduction
        if state['last_iteration'] >= self.max_iterations:
            return "iterations"
        if self.min_cost is None:
            return False
        cost = self._cost(state, frame.features, frame.target_mask)
        if cost <= self.min_cost:
            return "cost"
        return False

    def consolidate_features(self, state, frame):
        assert frame.did_reduction
        out = self._forward(state, frame.features)
        frame.consolidated_features = OrderedDict()
        frame.consolidated_features['single'] = out
        # prediction from single feature is simple:
        frame.prediction_mask = out

    def store_frame(self, state, key, frame, weight=1.0):
        logger.info(
            "Storing frame '%s' for consolidator update with weight %0.3f.", key, weight)
        target_mask = frame.target_mask
        stored = {
            'features': self._concat_features(frame.features),
            'target': target_mask,
            'weight': weight,
        }
        # special case in which we only keep the original frame. No update of
        # store needed here:
        if self.update_frame_store_size == 1 and self.update_keep_initial_frame:
            return
        # drop old frames if space is needed:
        # (use -1, because we store frames after update, so we have one frame more, actually)
        while len(state['stored_frames']) >= self.update_frame_store_size - 1:
            if self.update_keep_initial_frame:
                assert self.update_frame_store_size > 1
                del_key = list(state['stored_frames'])[1]
            else:
                assert self.update_frame_store_size > 0
                del_key = list(state['stored_frames'])[0]
            logger.info("Dropping frame %s from store", del_key)
            del state['stored_frames'][del_key]
        # store frame
        state['stored_frames'][key] = stored

    def update(self, state, frame, weight=1.0, steps=1):
        num_samples = len(state['stored_frames']) + 1
        logger.info(
            "Updating consolidator with %d samples, %d steps", num_samples, steps)
        #
        features = self._concat_features(frame.features)
        input_shape = list(np.shape(features))
        input_shape[0] = num_samples
        input_data = np.empty(shape=input_shape)
        #
        target_mask = frame.target_mask
        target_shape = list(np.shape(target_mask))
        target_shape[0] = num_samples
        target_data = np.empty(shape=target_shape)
        #
        weight_shape = [num_samples, ]
        weight_data = np.empty(shape=weight_shape)
        #
        input_data[0] = features
        target_data[0] = target_mask
        weight_data[0] = weight
        for n, value in enumerate(state['stored_frames'].values()):
            input_data[n + 1] = value['features']
            target_data[n + 1] = value['target']
            weight_data[n + 1] = value['weight']
        logger.info("Update weights: %s", weight_data)
        # train it
        for _ in range(steps):
            ret = state['net'].train(
                input_data=input_data, target_data=target_data, target_weight=weight_data)
        return ret

from collections import OrderedDict

import numpy as np
import tensorflow as tf
import logging

from ..BuiltNet import BuiltNet
from .FeatureSelector import FeatureSelector

logger = logging.getLogger(__name__)


class NetSelector(FeatureSelector):
    MAX_ITERATIONS = 50
    MIN_COST = 0.1

    def __init__(self):
        pass

    def configure(self, configuration):
        self.dtype = tf.float32
        # read config file for selector:
        sconf = configuration['selector']
        self.max_iterations = sconf['max_iterations']
        self.min_cost = sconf['min_cost']
        # read feature configuration:
        self.feature_counts = OrderedDict()
        for name, num in configuration['features']:
            self.feature_counts[name] = num
        #
        self.net_configuration = sconf['net']

    def setup(self, session):
        self.session = session

    def setup_tracking(self, state, output_features):
        nets = OrderedDict()
        # create a SelNet for every output layer that needs to be processed:
        for name, feature in output_features.items():
            logger.info("Building SelNet for feature %s", name)
            net = BuiltNet(
                self.session,
                self.net_configuration,
                input_shape=feature.get_shape().as_list(),
                use_input_variable=True,
                use_target_variable=True,
            )
            net.add_gradient()
            # net.initialize_variables()
            nets[name] = net
        state['nets'] = nets

    def load_data_for_selection(self, state, frame):
        assert frame.target_mask is not None
        for name, net in state['nets'].items():
            logger.info("Loading data for selection for %s", name)
            net.set_input(frame.features[name])
            net.set_target(frame.target_mask)

    def cost(self, state):
        costs = OrderedDict()
        for name, net in state['nets'].items():
            costs[name] = net.cost()
        return costs

    def forward(self, state):
        forwards = OrderedDict()
        for name, net in state['nets'].items():
            forwards[name] = net.forward()
        return forwards

    def calculate_forward(self, state):
        f = self.forward(state)
        state['forwards'] = f

    def start_training(self, state, frame):
        self.load_data_for_selection(state, frame)
        state['costs'] = self.cost(state)
        state['last_iteration'] = 0

    def training_step(self, state):
        state['last_iteration'] += 1
        for name, net in state['nets'].items():
            if self.min_cost is None:
                reached = False
            else:
                reached = state['costs'][name] <= self.min_cost
            if not reached:
                net.train()
        # recalculate costs:
        state['costs'] = self.cost(state)

    def training_done(self, state):
        if state['last_iteration'] >= self.max_iterations:
            return "iterations"
        cost_reached = False
        if self.min_cost is not None:
            for cost in state['costs'].values():
                if cost > self.min_cost:
                    cost_reached = False
                    break
                else:
                    cost_reached = True
        if cost_reached:
            return "cost"
        return False

    def training_costs_string(self, state):
        parts = []
        for name, cost in state['costs'].items():
            done = ""
            if self.min_cost is not None:
                if cost <= self.min_cost:
                    done = "*"
            parts.append("%s=%s%f" % (name, done, cost))
        start = "[Iteration %03d/%03d] " % (
            state['last_iteration'], self.max_iterations)
        return start + (", ".join(parts))

    def evaluate_selection(self, state, features, target_mask):
        logger.info("Evaluating feature impact")
        self.calculate_forward(state)
        feature_ratings = OrderedDict()
        feature_orders = OrderedDict()
        for name, net in state['nets'].items():
            input_data = features[name][0]
            forward = state['forwards'][name]
            diff1 = forward - target_mask
            net.set_target(diff1)
            in_diff1 = net.gradient()[0][0]
            diff2 = np.ones_like(target_mask)
            net.set_target(diff2)
            in_diff2 = net.gradient()[0][0]
            sal = (in_diff1 * input_data) + \
                (0.5 * in_diff2 * input_data * input_data)
            import pickle
            with open('/tmp/' + name + '.sal.p', 'wb') as f:
                pickle.dump(sal, f)
            with open('/tmp/' + name + '.feat.p', 'wb') as f:
                pickle.dump(input_data, f)
            with open('/tmp/' + name + '.indiff1.p', 'wb') as f:
                pickle.dump(in_diff1, f)
            with open('/tmp/' + name + '.indiff2.p', 'wb') as f:
                pickle.dump(in_diff2, f)
            with open('/tmp/' + name + '.target.p', 'wb') as f:
                pickle.dump(target_mask, f)
            with open('/tmp/' + name + '.forward.p', 'wb') as f:
                pickle.dump(forward, f)
            rating = np.sum(sal, (0, 1))
            order = np.argsort(rating)
            feature_ratings[name] = rating
            feature_orders[name] = order
        # store in state:
        state['feature_ratings'] = feature_ratings
        state['feature_orders'] = feature_orders
        #print(feature_ratings, feature_orders)

    def free_selection_nets(self, state):
        del state['nets']
        del state['forwards']
        del state['costs']

    def _feature_reduce(self, data, order, num):
        sh = data.shape
        r = np.zeros((1, sh[1], sh[2], num), dtype=data.dtype)
        for n in range(num):
            r[0, :, :, n] = data[0, :, :, order[n]]
        return r

    def reduce_features(self, state, features):
        reduced = OrderedDict()
        for name, orders in state['feature_orders'].items():
            reduced[name] = self._feature_reduce(
                features[name], orders, self.feature_counts[name])
        return reduced
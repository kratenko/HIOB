import logging

import numpy as np
import tensorflow as tf

import hiob.base
from collections import OrderedDict
from hiob.netbuilder import BuiltNet

logger = logging.getLogger(__name__)


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


class FeatureSelector(hiob.base.HiobModule):

    def reduce_features(self, tracking, frame):
        raise NotImplementedError()


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
            print(np.shape(sal))
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
        print(feature_ratings, feature_orders)

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


class SelectNet(FeatureSelector):
    DROPOUT_KEEP = 0.7
    LEARNING_RATE = 0.000001

    def __init__(self):
        pass

    def configure(self, configuration):
        self.dtype = tf.float32
        self.input_shape = (1, 46, 46, 512)  # (1, 46, 46, 512)
        self.target_shape = self.input_shape[:3] + (1, )  # (1, 46, 46, 1)

        self.dropout_keep = self.DROPOUT_KEEP
        self.learning_rate = self.LEARNING_RATE

    def setup(self, session):
        self.session = session

    def setup_tracking(self, state, output_features):
        input_initial = tf.zeros(
            self.input_shape,
            name="input_initial",
            dtype=self.dtype,
        )
        state['input_variable'] = tf.Variable(
            initial_value=input_initial,
            name="input_variable",
            dtype=self.dtype,
            trainable=False,
        )
        state['input_placeholder'] = tf.placeholder(
            self.dtype, self.input_shape, 'input_placeholder')
        state['input_assign'] = state['input_variable'].assign(
            state['input_placeholder'])
        state['target_placeholder'] = tf.placeholder(
            self.dtype, self.target_shape, 'target_placeholder')

        # create network:
        state['conv1_weight_variable'] = weight_variable(
            (3, 3, self.input_shape[3], 1), 'conv1_weight_variable', 1e-7)
        state['conv1_bias_variable'] = bias_variable(
            (1,), 'conv1_bias_variable', value=0)
        # dropout layer:
        state['dropout'] = tf.nn.dropout(
            state['input_variable'],
            keep_prob=self.dropout_keep,
            name="dropout",
        )
        state['conv1'] = tf.nn.conv2d(state['dropout'], state['conv1_weight_variable'], strides=[
            1, 1, 1, 1], padding="SAME", name="conv1") + state['conv1_bias_variable']

        # cost function
        state['cost_function'] = tf.reduce_mean(
            tf.square(state['conv1'] - state['target_placeholder']))
        state['optimizer'] = tf.train.AdamOptimizer(
            self.learning_rate, name="optimizer")
        state['trainer'] = state['optimizer'].minimize(
            state['cost_function'], name="trainer")
        state['gradient_function'] = state['optimizer'].compute_gradients(
            state['cost_function'], [state['input_variable']])

    def cost(self, state, frame):
        feed_dict = {
            state['input_placeholder']: frame.features,
            state['target_placeholder']: frame.target_mask,
        }
        return self.session.run([self.cost_function], feed_dict=feed_dict)[0]

    def forward(self, state, frame):
        feed_dict = {
            state['input_placeholder']: frame.features,
            state['target_placeholder']: frame.target_mask,
        }
        return self.session.run([self.conv1], feed_dict=feed_dict)[0]

    def train(self, state, frame):
        feed_dict = {
            state['input_placeholder']: frame.features,
            state['target_placeholder']: frame.target_mask,
        }
        self.session.run([self.train], feed_dict=feed_dict)

    def gradient(self, state, frame):
        feed_dict = {
            state['input_placeholder']: frame.features,
            state['target_placeholder']: frame.target_mask,
        }
        vals = self.session.run(
            [gr for gr, _ in self.gradient_function], feed_dict=feed_dict)
        return vals


class SelNet(object):
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

import tensorflow as tf

from .FeatureSelector import FeatureSelector
from .util import weight_variable, bias_variable


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
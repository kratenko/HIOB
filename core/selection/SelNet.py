import tensorflow as tf
import logging

from .util import weight_variable, bias_variable

logger = logging.getLogger(__name__)


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
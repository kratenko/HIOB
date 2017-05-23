"""
Created on 2016-11-23

@author: Peer Springstübe
"""

import logging

import tensorflow as tf
import numpy as np

logger = logging.getLogger(__name__)


class BuiltNet(object):
    _LAST_SERIAL = 0

    def __init__(self, session, conf, input_shape=None, use_input_variable=False, use_target_variable=False):
        BuiltNet._LAST_SERIAL += 1
        self.serial = BuiltNet._LAST_SERIAL

        if 'name' in conf:
            self.name = conf['name']
        else:
            self.name = "BuiltNet#{}".format(self.serial)

        self._last_layer_serial = 0
        self.dtype = tf.float32
        self.session = session
        self.variables = []

        with tf.name_scope(self.name):
            self.build_net(
                conf,
                input_shape,
                use_input_variable,
                use_target_variable,
            )

    def build_net(self, conf, input_shape=None, use_input_variable=False, use_target_variable=False):
        if input_shape is None:
            self.input_shape = conf['input_shape']
        else:
            self.input_shape = input_shape

        # create net input (placeholder and maybe variable):
        self.input_placeholder = tf.placeholder(
            self.dtype,
            self.input_shape,
            name='input_placeholder',
        )
        if use_input_variable:
            self.input_initial = tf.zeros(
                shape=self.input_shape,
                dtype=self.dtype,
                name='input_initializer',
            )
            self.input_variable = tf.Variable(
                self.input_initial,
                trainable=False,
                name='input_variable',
            )
            self.variables.append(self.input_variable)
            self.input_assign = self.input_variable.assign(
                self.input_placeholder)
            last_layer = self.input_variable
        else:
            self.input_variable = None
            last_layer = self.input_placeholder

        # build the net layers:
        layers = []
        for layer_def in conf['layers']:
            layer = self.build_layer(last_layer, layer_def)
            layers.append(layer)
            last_layer = layer
        self.output_layer = last_layer

        # target stuff:
        self.target_placeholder = tf.placeholder(
            self.dtype,
            shape=self.output_layer.get_shape(),
            name="target_placeholder",
        )
        self.target_weight_shape = (
            self.target_placeholder.get_shape().as_list()[0]
        )
        self.target_weight_placeholder = tf.placeholder(
            self.dtype,
            shape=self.target_weight_shape,
            name='target_weight_placeholder',
        )
        if use_target_variable:
            self.target_initial = tf.zeros(
                shape=self.target_placeholder.get_shape(),
                dtype=self.dtype,
                name='target_initializer',
            )
            self.target_variable = tf.Variable(
                self.target_initial,
                dtype=self.dtype,
                name='target_variable',
                trainable=False,
            )
            self.variables.append(self.target_variable)
            self.target_assign = self.target_variable.assign(
                self.target_placeholder)
            self.target_weight_initial = tf.ones(
                shape=self.target_weight_shape,
                dtype=self.dtype,
                name='target_weight_initializer',
            )
            self.target_weight_variable = tf.Variable(
                self.target_weight_initial,
                dtype=self.dtype,
                name='target_weight_variable',
                trainable=False,
            )
            self.variables.append(self.target_weight_variable)
            self.target_weight_assign = self.target_weight_variable.assign(
                self.target_weight_placeholder)
        else:
            self.target_variable = None
            self.target_weight_variable = None

        self.cost_function = self.build_cost_function(conf['cost'])
        self.optimizer = self.build_optimizer(conf['optimizer'])
        self.trainer = self.optimizer.minimize(
            self.cost_function, name='trainer')

        #
        self.gradient_function = None

    def build_layer(self, input_layer, conf):
        # increase serial number for automated layer naming:
        self._last_layer_serial += 1
        logger.info("Layer %d", self._last_layer_serial)
        # create layer of certain type:
        if conf['type'] == 'dropout':
            layer = self.build_dropout_layer(input_layer, conf)
        elif conf['type'] == 'conv':
            layer = self.build_conv_layer(input_layer, conf)
        elif conf['type'] == 'activation':
            layer = self.build_activation_layer(input_layer, conf)
        else:
            raise ValueError("Unknown layer type: " + conf['type'])
        logger.info(
            "Layer %d shape: (%s)", self._last_layer_serial, layer.get_shape().as_list())
        return layer

    def build_dropout_layer(self, input_layer, conf):
        if 'name' in conf:
            name = conf['name']
        else:
            name = "dropout{}".format(self._last_layer_serial)
        return tf.nn.dropout(
            input_layer,
            keep_prob=conf['keep_prob'],
            name=name,
        )

    def build_activation_layer(self, input_layer, conf):
        if 'name' in conf:
            name = conf['name']
        else:
            name = "activation{}".format(self._last_layer_serial)
        if conf['function'] == 'relu':
            return tf.nn.relu(input_layer, name)
        elif conf['function'] == 'crelu':
            crelu = tf.nn.crelu(input_layer, name)
            # tensorflow bug, see http://stackoverflow.com/q/40852397/1358283
            [b, nx, ny, nz] = input_layer.get_shape().as_list()
            crelu.set_shape([b, nx, ny, 2 * nz])
            return crelu
        elif conf['function'] == 'maxout':
            return self.maxout(input_layer)
        else:
            raise ValueError(
                "Unknown activation function: " + conf['function'])

    def maxout(self, inputs, num_units, axis=None):
        # taken from http://stackoverflow.com/a/40537068/1358283
        # not working...
        shape = inputs.get_shape().as_list()
        if axis is None:
            # Assume that channel is the last dimension
            axis = -1
        num_channels = shape[axis]
        if num_channels % num_units:
            raise ValueError('number of features({}) is not a multiple of num_units({})'
                             .format(num_channels, num_units))
        shape[axis] = -1
        shape += [num_channels // num_units]
        outputs = tf.reduce_max(tf.reshape(inputs, shape), -1, keep_dims=False)
        return outputs

    def build_conv_layer(self, input_layer, conf):
        if 'name' in conf:
            layer_name = conf['name']
        else:
            layer_name = "conv{}".format(self._last_layer_serial)
        # calculate shape of conv layer:
        kernel_size = conf['kernel_size']
        channels = conf['channels']
        in_shape = input_layer.get_shape().as_list()
        weight_shape = (kernel_size, kernel_size, in_shape[3], channels)
        # create variables, weight and bias:
        weight_initial = self.build_initial(
            weight_shape,
            conf['weight_initial'],
            name=layer_name + '_weight_initial',
        )
        weight_variable = tf.Variable(
            initial_value=weight_initial,
            trainable=True,
            dtype=self.dtype,
            name=layer_name + '_weight_variable',
        )
        bias_shape = (channels,)
        bias_initial = self.build_initial(
            bias_shape,
            conf['bias_initial'],
            name=layer_name + '_bias_initial',
        )
        bias_variable = tf.Variable(
            initial_value=bias_initial,
            trainable=True,
            dtype=self.dtype,
            name=layer_name + '_bias_variable',
        )
        # create conv layer:

        conv = tf.nn.conv2d(
            input_layer,
            weight_variable,
            strides=[1, 1, 1, 1],
            padding='SAME',
            name=layer_name + '_conv',
        )
        #bias = conv + bias_variable
        bias = tf.add(
            conv, bias_variable, name=layer_name + '_bias_add')
        #nonlin = tf.nn.softplus(bias, name=layer_name + '_nonlin')
        # remember variables:
        self.variables.extend([weight_variable, bias_variable])
        return bias
        # return nonlin

    def build_initial(self, shape, conf, name=None):
        # prepare configuration:
        if type(conf) == str:
            name = conf
            parms = {}
        elif type(conf) == int or type(conf) == float:
            name = 'constant'
            parms = {'value': float(conf)}
        else:
            name = conf[0]
            if len(conf) > 1:
                parms = conf[1]
            else:
                parms = {}
        # copy dict before updating it:
        parms = dict(parms)
        if 'name' not in parms:
            # name from conf file. If not there use created one:
            if name is not None:
                parms['name'] = name
        # if 'dtype' not in parms:
        #    parms['dtype'] = self.dtype
        # create initial:
        if name == 'truncated_normal':
            logger.info("Shape %s", shape)
            logger.info("Parms %s", parms)
            return tf.truncated_normal(shape, **parms)
        elif name == 'zeros':
            return tf.zeros(shape, **parms)
        elif name == 'constant':
            return tf.fill(shape, **parms)
        else:
            raise ValueError("Unknown initial: " + name)

    def build_cost_function(self, conf):
        target = self.target_variable or self.target_placeholder
        target_weight = self.target_weight_variable or self.target_weight_placeholder
        if conf == 'mean_square':
            return tf.reduce_mean(tf.square(self.output_layer - target)) * target_weight
        else:
            raise ValueError("Unknown cost function: " + conf)

    def build_optimizer(self, conf):
        name = conf[0]
        parms = conf[1]
        if name == 'adam':
            return tf.train.AdamOptimizer(**parms)
        elif name == 'momentum':
            return tf.train.MomentumOptimizer(**parms)
        else:
            raise ValueError("Unknown optimizer: " + name)

    def add_gradient(self):
        # gradient is only possible if we have an input variable
        assert self.input_variable is not None
        with tf.name_scope(self.name):
            self.gradient_function = self.optimizer.compute_gradients(
                self.cost_function, [self.input_variable])

    def initialize_variables(self):
        # TODO: optimizer is not initialized, yet!
        self.session.run(tf.initialize_variables(self.variables))

    # methods to work with net:

    def set_input(self, input_data):
        assert self.input_variable is not None
        self.session.run(
            [self.input_assign], feed_dict={self.input_placeholder: input_data})

    def set_target(self, target_data, target_weight=None):
        assert self.target_variable is not None
        self.session.run(
            [self.target_assign], feed_dict={self.target_placeholder: target_data})
        if target_weight is None:
            target_weight = np.ones(len(target_data))
        self.session.run(
            [self.target_weight_assign], feed_dict={self.target_weight_placeholder: target_weight})

    def train(self, input_data=None, target_data=None, target_weight=None):
        assert self.input_variable is None or input_data is None
        assert self.target_variable is None or target_data is None
        feed_dict = {}
        if self.input_variable is None:
            feed_dict[self.input_placeholder] = input_data
        if self.target_variable is None:
            feed_dict[self.target_placeholder] = target_data
            if target_weight is None:
                target_weight = np.ones(len(target_data))
            feed_dict[self.target_weight_placeholder] = target_weight
        # print("TRAIN")
        # for k, v in feed_dict.items():
        #    print(k, np.shape(v), np.average(v))
        self.session.run([self.trainer], feed_dict=feed_dict)

    def forward(self, input_data=None):
        assert self.input_variable is None or input_data is None
        if self.input_variable:
            return self.session.run([self.output_layer])[0]
        else:
            return self.session.run([self.output_layer], feed_dict={self.input_placeholder: input_data})[0]

    def cost(self, input_data=None, target_data=None, target_weight=None):
        assert self.input_variable is None or input_data is None
        assert self.target_variable is None or target_data is None
        feed_dict = {}
        if self.input_variable is None:
            feed_dict[self.input_placeholder] = input_data
        if self.target_variable is None:
            feed_dict[self.target_placeholder] = target_data
            if target_weight is None:
                target_weight = np.ones(len(target_data))
            feed_dict[self.target_weight_placeholder] = target_weight
        return self.session.run([self.cost_function], feed_dict=feed_dict)[0]

    def gradient(self):
        assert self.input_variable is not None
        vals = self.session.run([gr for gr, _ in self.gradient_function])
        return vals

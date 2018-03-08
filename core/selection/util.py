import tensorflow as tf


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
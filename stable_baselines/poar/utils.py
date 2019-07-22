import tensorflow as tf

from stable_baselines.a2c.utils import ortho_init, conv, linear, conv_to_fc
import numpy as np
from ipdb import set_trace as tt
from srl_zoo.utils import printYellow, printGreen, printRed

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5


def conv_t(input_tensor, scope, *, n_filters, filter_size, stride, output_shape=None,
         pad='VALID', init_scale=1.0, data_format='NHWC', one_dim_bias=False):
    """
    Creates a 2d convolutional layer for TensorFlow

    :param input_tensor: (TensorFlow Tensor) The input tensor for the convolution
    :param scope: (str) The TensorFlow variable scope
    :param n_filters: (int) The number of filters
    :param filter_size:  (Union[int, [int], tuple<int, int>]) The filter size for the squared kernel matrix,
    or the height and width of kernel filter if the input is a list or tuple
    :param stride: (int) The stride of the convolution
    :param pad: (str) The padding type ('VALID' or 'SAME')
    :param init_scale: (int) The initialization scale
    :param data_format: (str) The data format for the convolution weights
    :param one_dim_bias: (bool) If the bias should be one dimentional or not
    :return: (TensorFlow Tensor) 2d convolutional layer
    """
    if isinstance(filter_size, list) or isinstance(filter_size, tuple):
        assert len(filter_size) == 2, \
            "Filter size must have 2 elements (height, width), {} were given".format(len(filter_size))
        filter_height = filter_size[0]
        filter_width = filter_size[1]
    else:
        filter_height = filter_size
        filter_width = filter_size
    if data_format == 'NHWC':
        channel_ax = 3
        strides = [1, stride, stride, 1]
        bshape = [1, 1, 1, n_filters]
    elif data_format == 'NCHW':
        channel_ax = 1
        strides = [1, 1, stride, stride]
        bshape = [1, n_filters, 1, 1]
    else:
        raise NotImplementedError
    bias_var_shape = [n_filters] if one_dim_bias else [1, n_filters, 1, 1]
    n_input = input_tensor.get_shape()[channel_ax].value
    wshape = [filter_height, filter_width, n_filters, n_input]
    with tf.variable_scope(scope):
        weight = tf.get_variable("w", wshape, initializer=ortho_init(init_scale))
        bias = tf.get_variable("b", bias_var_shape, initializer=tf.constant_initializer(0.0))
        if not one_dim_bias and data_format == 'NHWC':
            bias = tf.reshape(bias, bshape)
        return bias + tf.nn.conv2d_transpose(input_tensor, weight, strides=strides, output_shape=output_shape,
                                             padding=pad, data_format=data_format)


def batch_norm(inputs, mode='NHWC'):
    if mode == 'NHWC':
        norm_func = tf.layers.BatchNormalization(axis=-1,momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON,
                                                    center=True, scale=True, fused=True)
    elif mode == 'NCHW':
        norm_func = tf.layers.BatchNormalization(axis=1,momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON,
                                                    center=True, scale=True, fused=True)
    else:
        raise NotImplementedError
    return norm_func(inputs)


def batch_norm_relu_deconv(input_tensor, scope, *, n_filters, filter_size, stride, output_shape=None,
                           pad='VALID', init_scale=1.0, data_format='NHWC', one_dim_bias=False):
    if isinstance(filter_size, list) or isinstance(filter_size, tuple):
        assert len(filter_size) == 2, \
            "Filter size must have 2 elements (height, width), {} were given".format(len(filter_size))
        filter_height = filter_size[0]
        filter_width = filter_size[1]
    else:
        filter_height = filter_size
        filter_width = filter_size
    channel_ax = 3
    strides = [1, stride, stride, 1]
    bshape = [1, 1, 1, n_filters]

    bias_var_shape = [n_filters] if one_dim_bias else [1, n_filters, 1, 1]
    n_input = input_tensor.get_shape()[channel_ax].value
    wshape = [filter_height, filter_width, n_filters, n_input]

    with tf.variable_scope(scope):
        weight = tf.get_variable("w", wshape, initializer=ortho_init(init_scale))
        bias = tf.get_variable("b", bias_var_shape, initializer=tf.constant_initializer(0.0))
        if not one_dim_bias and data_format == 'NHWC':
            bias = tf.reshape(bias, bshape)
        d_conv_layer = bias + tf.nn.conv2d_transpose(input_tensor, weight, strides=strides, output_shape=output_shape,
                                             padding=pad, data_format=data_format)

        return tf.nn.relu(tf.compat.v1.layers.batch_normalization(d_conv_layer, axis=-1, momentum=_BATCH_NORM_DECAY,
                                                                  epsilon=_BATCH_NORM_EPSILON,
                                                                  center=True, scale=True, fused=True))


def bn_autoencoder(obs, state_dim):
    activation = tf.nn.relu
    e1 = activation(batch_norm(conv(scope='e1', input_tensor=obs, n_filters=64, filter_size=8, stride=4, pad='SAME')))
    e2 = activation(batch_norm(conv(scope='e2', input_tensor=e1, n_filters=64, filter_size=4, stride=2, pad='SAME')))
    e3 = activation(batch_norm(conv(scope='e3', input_tensor=e2, n_filters=64, filter_size=4, stride=2, pad='SAME')))

#    m3 = tf.layers.max_pooling2d(e3, pool_size=2, strides=2)
    m3_flat = conv_to_fc(e3)
    latent = linear(scope='latent', input_tensor=m3_flat, n_hidden=state_dim)

    d0 = tf.reshape(linear(scope='deocder_fc', input_tensor=latent, n_hidden=m3_flat.get_shape()[-1]), shape=tf.shape(e3))
    d1 = activation(batch_norm(tf.layers.conv2d_transpose(d0, filters=64, kernel_size=4, strides=2, padding='same')))
    d2 = activation(batch_norm(tf.layers.conv2d_transpose(d1, filters=64, kernel_size=4, strides=2, padding='same')))
    d3 = activation(batch_norm(tf.layers.conv2d_transpose(d2, filters=64, kernel_size=8, strides=4, padding='same')))
    output = tf.nn.sigmoid(conv(scope='reconstruction', input_tensor=d3, n_filters=3, filter_size=3, stride=1, pad='SAME'))
    return output, latent

def autoencoderMP(obs, state_dim):
    activation = tf.nn.relu
    e1 = activation(batch_norm(conv(scope='e1', input_tensor=obs, n_filters=64, filter_size=3, stride=1, pad='SAME')))
    m1 = tf.layers.max_pooling2d(e1, pool_size=2, strides=2)
    e2 = activation(batch_norm(conv(scope='e2', input_tensor=m1, n_filters=64, filter_size=3, stride=1, pad='SAME')))
    m2 = tf.layers.max_pooling2d(e2, pool_size=2, strides=2)
    e3 = activation(batch_norm(conv(scope='e3', input_tensor=m2, n_filters=128, filter_size=3, stride=1, pad='SAME')))

    m3 = tf.layers.max_pooling2d(e3, pool_size=2, strides=2)
    m3_flat = conv_to_fc(m3)
    latent = linear(scope='latent', input_tensor=m3_flat, n_hidden=state_dim)

    d0 =tf.reshape(linear(scope='deocder_fc', input_tensor=latent, n_hidden=m3_flat.get_shape()[-1]), shape=tf.shape(m3))
    d1 = activation(batch_norm(tf.layers.conv2d_transpose(d0, filters=128, keroonel_size=3, strides=2, padding='same')))
    d2 = activation(batch_norm(tf.layers.conv2d_transpose(d1, filters=64, kernel_size=3, strides=2, padding='same')))
    d3 = activation(batch_norm(tf.layers.conv2d_transpose(d2, filters=64, kernel_size=3, strides=2, padding='same')))
    output = tf.nn.sigmoid(conv(scope='reconstruction', input_tensor=d3, n_filters=3, filter_size=3, stride=1, pad='SAME'))
    return output, latent

def nature_autoencoder(obs, state_dim, name=''):
    activation = tf.nn.relu
    e1 = activation(conv(scope=name+'e1', input_tensor=obs, n_filters=64, filter_size=8, stride=4, pad='SAME'))
    e2 = activation(conv(scope=name+'e2', input_tensor=e1, n_filters=64, filter_size=4, stride=2, pad='SAME'))
    e3 = activation(conv(scope=name+'e3', input_tensor=e2, n_filters=64, filter_size=3, stride=1, pad='SAME'))

    #m3 = tf.layers.max_pooling2d(e3, pool_size=2, strides=2)
    m3_flat = conv_to_fc(e3)
    latent = linear(scope=name+'latent', input_tensor=m3_flat, n_hidden=state_dim)

    d0 =tf.reshape(linear(scope=name+'deocder_fc', input_tensor=latent, n_hidden=m3_flat.get_shape()[-1]), shape=tf.shape(e3))
    d1 = activation(batch_norm(tf.layers.conv2d_transpose(d0, filters=64, kernel_size=3, strides=1, padding='same')))
    d2 = activation(batch_norm(tf.layers.conv2d_transpose(d1, filters=64, kernel_size=4, strides=2, padding='same')))
    d3 = activation(batch_norm(tf.layers.conv2d_transpose(d2, filters=64, kernel_size=8, strides=4, padding='same')))
    output = tf.nn.sigmoid(conv(scope=name+'reconstruction', input_tensor=d3, n_filters=3, filter_size=3, stride=1, pad='SAME'))
    return output, latent


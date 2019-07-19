import tensorflow as tf

from tensorflow import nn
from stable_baselines.a2c.utils import ortho_init, conv, linear, conv_to_fc
from tensorflow import keras
from ipdb import set_trace as tt

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



def batch_norm(inputs):
    return tf.compat.v1.layers.batch_normalization(
        inputs=inputs, axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON,
        center=True, scale=True, fused=True)


def batch_norm_relu_conv(input_tensor, scope, *, n_filters, filter_size, stride, pad='VALID', init_scale=1.0,
                         one_dim_bias=False):
    if isinstance(filter_size, list) or isinstance(filter_size, tuple):
        assert len(filter_size) == 2, \
            "Filter size must have 2 elements (height, width), {} were given".format(len(filter_size))
        filter_height = filter_size[0]
        filter_width = filter_size[1]
    else:
        filter_height = filter_size
        filter_width = filter_size

    bias_var_shape = [n_filters] if one_dim_bias else [1, n_filters, 1, 1]
    n_input = input_tensor.get_shape()[3].value
    wshape = [filter_height, filter_width, n_input, n_filters]
    with tf.variable_scope(scope):
        weight = tf.get_variable("w", wshape, initializer=ortho_init(init_scale))
        bias = tf.get_variable("b", bias_var_shape, initializer=tf.constant_initializer(0.0))
        conv_layer = bias + tf.nn.conv2d(input_tensor, weight, strides=[1, 1, stride, stride], padding=pad, data_format='NHWC')
        return conv_layer
        # if data_format == 'NHWC':
        #     return tf.nn.relu(tf.compat.v1.layers.batch_normalization(conv_layer, axis=-1, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON,
        #                                     center=True, scale=True, fused=True))
        # else:
        #     return tf.nn.relu(keras.layers.BatchNormalization(axis=1, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON,
        #                                     center=True, scale=True, fused=True)(conv_layer))


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
        d_conv_layer = bias + tf.nn.conv2d_transpose(input_tensor, weight, strides=strides, output_shape=output_shape,
                                             padding=pad, data_format=data_format)
        if data_format == 'NHWC':
            return tf.nn.relu(tf.compat.v1.layers.batch_normalization(d_conv_layer, axis=-1, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON,
                                            center=True, scale=True, fused=True))
        else:
            return tf.nn.relu(keras.layers.BatchNormalization(axis=1, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON,
                                            center=True, scale=True, fused=True)(d_conv_layer))



def cnn_autoencoder0(obs_shape, obs, state_dim=512, **kwargs):
    # e1 = tf.keras.layers.Conv2D(filters=64, kernel_size=7, strides=4, padding='valid', activation='relu')(obs)
    # e2 = tf.keras.layers.Conv2D(filters=64, kernel_size=5, strides=2, padding='valid', activation='relu')(e1)

    e1 = tf.nn.relu(tf.keras.layers.BatchNormalization()(tf.layers.conv2d(obs, filters=64, kernel_size=8, strides=4,padding='same')))
    e2 = tf.nn.relu(tf.keras.layers.BatchNormalization()(tf.layers.conv2d(e1, filters=64, kernel_size=5, strides=2)))
    e3 = tf.nn.relu(tf.keras.layers.BatchNormalization()(tf.layers.conv2d(e2, filters=64, kernel_size=5, strides=2)))

    d1 = tf.nn.relu(
        tf.keras.layers.BatchNormalization()(tf.layers.conv2d_transpose(e3,filters=64, kernel_size=6, strides=2)))
    d2 = tf.nn.relu(
        tf.keras.layers.BatchNormalization()(tf.layers.conv2d_transpose(d1, filters=64, kernel_size=6, strides=2)))
    d3 = tf.nn.relu(
        tf.keras.layers.BatchNormalization()(tf.layers.conv2d_transpose(d2, filters=64, kernel_size=7, strides=4)))
    out = tf.nn.softmax(tf.layers.conv2d(d3, filters=3, kernel_size=4, strides=1, padding='valid'))
    # d1 = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=5, strides=2, activation='relu')(e2)
    # d2 = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=7, strides=4, activation='relu')(d1)
    # d3 = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=4, strides=1, activation='relu')(d2)
    # d4 = tf.keras.layers.Conv2D(filters=3, kernel_size=3, strides=1, padding='valid', activation='relu')(d3)
    return out, tf.layers.flatten(e3)

def cnn_autoencoder2(obs_shape, obs, state_dim=512, **kwargs):

    e1 = batch_norm_relu_conv(obs, 'e1', n_filters=64, filter_size=8, stride=4, init_scale=np.square(2), **kwargs)
    e2 = batch_norm_relu_conv(e1, 'e2', n_filters=64, filter_size=4, stride=2, init_scale=np.square(2), **kwargs)
    e3 = batch_norm_relu_conv(e2, 'e3', n_filters=64, filter_size=3, stride=1, init_scale=np.square(2), **kwargs)
    x = conv_to_fc(e3)

    d1 = batch_norm_relu_deconv(e3, 'd1', n_filters=64, filter_size=3, stride=1, output_shape=tf.shape(e2),
                    init_scale=np.square(2), **kwargs)
    d2 = batch_norm_relu_deconv(d1, 'd2', n_filters=64, filter_size=4, stride=2, output_shape=tf.shape(e1),
                    init_scale=np.square(2), **kwargs)
    d3 = batch_norm_relu_deconv(d2, 'd3', n_filters=3, filter_size=8, stride=4, output_shape=obs_shape,
                                init_scale=np.square(2), **kwargs)

    return tf.nn.softmax(conv(d3, 'd4', n_filters=3, filter_size=3, stride=1, pad="SAME",
                              init_scale=np.square(2), **kwargs)), x
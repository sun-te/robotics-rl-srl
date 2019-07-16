import warnings
from itertools import zip_longest

import numpy as np
import tensorflow as tf

from stable_baselines.a2c.utils import conv, linear, conv_to_fc, batch_to_seq, seq_to_batch, lstm
from stable_baselines.common.policies import ActorCriticPolicy, RecurrentActorCriticPolicy
from stable_baselines.poar.utils import conv_t
from srl_zoo.utils import printYellow, printGreen, printRed
from ipdb import set_trace as tt

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5


def nature_cnn(scaled_images, **kwargs):
    """
    CNN from Nature paper.

    :param scaled_images: (TensorFlow Tensor) Image input placeholder
    :param kwargs: (dict) Extra keywords parameters for the convolutional layers of the CNN
    :return: (TensorFlow Tensor) The CNN output layer
    """
    activ = tf.nn.relu
    layer_1 = activ(conv(scaled_images, 'c1', n_filters=32, filter_size=8, stride=4, init_scale=np.sqrt(2), **kwargs))
    layer_2 = activ(conv(layer_1, 'c2', n_filters=64, filter_size=4, stride=2, init_scale=np.sqrt(2), **kwargs))
    layer_3 = activ(conv(layer_2, 'c3', n_filters=64, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_3 = conv_to_fc(layer_3)
    return activ(linear(layer_3, 'fc1', n_hidden=512, init_scale=np.sqrt(2)))


def batch_norm(inputs):
    return tf.compat.v1.layers.batch_normalization(
        inputs=inputs, axis=3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON,
        center=True, scale=True, fused=True)


class AutoEncoder:
    def __init__(self):
        pass

    def encode(self, obs):
        pass

    def decode(self, latent_x):
        pass

    def forward(self, obs):
        return tf.reshape(self.decode(self.encode(obs)), obs.shape)


class CNNAutoEncoer(AutoEncoder):
    def __init__(self, state_dim=512):
        super(CNNAutoEncoer, self).__init__()
        self.state_dim = state_dim
        self.hidden_size = None

    def encode(self, obs, **kwargs):
        relu = tf.nn.relu
        layer1 = conv(obs, 'e1', n_filters=64, filter_size=8, stride=4, init_scale=np.square(2), **kwargs)
        layer1 = relu(batch_norm(layer1))  # (?, 55, 55, 64)

        layer2 = conv(layer1, 'e2', n_filters=64, filter_size=4, stride=2, init_scale=np.square(2), **kwargs)
        layer2 = relu(batch_norm(layer2))  # (?, 26, 26, 64)

        layer3 = conv(layer2, 'e3', n_filters=64, filter_size=3, stride=1, init_scale=np.square(2), **kwargs)
        layer3 = relu(batch_norm(layer3))  # (?, 24, 24, 64)

        layer4 = conv_to_fc(layer3)
        layers = [layer3, layer2, layer1, obs]

        self.hidden_size = [layer4.get_shape()[1]] + [tf.shape(arr) for arr in layers]

        return relu(linear(layer4, 'e_fc', n_hidden=self.state_dim, init_scale=np.sqrt(2)))

    def decode(self, x, **kwargs):
        relu = tf.nn.relu
        layer1 = relu(linear(x, 'd_fc', n_hidden=self.hidden_size[0], init_scale=np.sqrt(2)))
        layer1 = tf.reshape(layer1, self.hidden_size[1])

        layer2 = conv_t(layer1, 'd1', n_filters=64, filter_size=3, stride=1, output_shape=self.hidden_size[2],
                        init_scale=np.square(2), **kwargs)
        layer2 = relu(batch_norm(layer2))

        layer3 = conv_t(layer2, 'd2', n_filters=64, filter_size=4, stride=2, output_shape=self.hidden_size[3],
                        init_scale=np.square(2), **kwargs)
        layer3 = relu(batch_norm(layer3))

        layer4 = conv_t(layer3, 'd3', n_filters=3, filter_size=8, stride=4, output_shape=self.hidden_size[4],
                        init_scale=np.square(2), **kwargs)
        return tf.nn.sigmoid(conv(layer4, scope='d4', n_filters=3, filter_size=3, stride=1, pad='SAME',
                               init_scale=np.square(2),  **kwargs))

    def forward(self, obs, **kwargs):
        latent_obs = self.encode(obs, **kwargs)
        return self.decode(latent_obs, **kwargs)


def mlp_extractor(flat_observations, net_arch, act_fun):
    """
    Constructs an MLP that receives observations as an input and outputs a latent representation for the policy and
    a value network. The ``net_arch`` parameter allows to specify the amount and size of the hidden layers and how many
    of them are shared between the policy network and the value network. It is assumed to be a list with the following
    structure:

    1. An arbitrary length (zero allowed) number of integers each specifying the number of units in a shared layer.
       If the number of ints is zero, there will be no shared layers.
    2. An optional dict, to specify the following non-shared layers for the value network and the policy network.
       It is formatted like ``dict(vf=[<value layer sizes>], pi=[<policy layer sizes>])``.
       If it is missing any of the keys (pi or vf), no non-shared layers (empty list) is assumed.

    For example to construct a network with one shared layer of size 55 followed by two non-shared layers for the value
    network of size 255 and a single non-shared layer of size 128 for the policy network, the following layers_spec
    would be used: ``[55, dict(vf=[255, 255], pi=[128])]``. A simple shared network topology with two layers of size 128
    would be specified as [128, 128].

    :param flat_observations: (tf.Tensor) The observations to base policy and value function on.
    :param net_arch: ([int or dict]) The specification of the policy and value networks.
        See above for details on its formatting.
    :param act_fun: (tf function) The activation function to use for the networks.
    :return: (tf.Tensor, tf.Tensor) latent_policy, latent_value of the specified network.
        If all layers are shared, then ``latent_policy == latent_value``
    """
    latent = flat_observations
    policy_only_layers = []  # Layer sizes of the network that only belongs to the policy network
    value_only_layers = []  # Layer sizes of the network that only belongs to the value network

    # Iterate through the shared layers and build the shared parts of the network
    for idx, layer in enumerate(net_arch):
        if isinstance(layer, int):  # Check that this is a shared layer
            layer_size = layer
            latent = act_fun(linear(latent, "shared_fc{}".format(idx), layer_size, init_scale=np.sqrt(2)))
        else:
            assert isinstance(layer, dict), "Error: the net_arch list can only contain ints and dicts"
            if 'pi' in layer:
                assert isinstance(layer['pi'], list), "Error: net_arch[-1]['pi'] must contain a list of integers."
                policy_only_layers = layer['pi']

            if 'vf' in layer:
                assert isinstance(layer['vf'], list), "Error: net_arch[-1]['vf'] must contain a list of integers."
                value_only_layers = layer['vf']
            break  # From here on the network splits up in policy and value network

    # Build the non-shared part of the network
    latent_policy = latent
    latent_value = latent
    for idx, (pi_layer_size, vf_layer_size) in enumerate(zip_longest(policy_only_layers, value_only_layers)):
        if pi_layer_size is not None:
            assert isinstance(pi_layer_size, int), "Error: net_arch[-1]['pi'] must only contain integers."
            latent_policy = act_fun(linear(latent_policy, "pi_fc{}".format(idx), pi_layer_size, init_scale=np.sqrt(2)))

        if vf_layer_size is not None:
            assert isinstance(vf_layer_size, int), "Error: net_arch[-1]['vf'] must only contain integers."
            latent_value = act_fun(linear(latent_value, "vf_fc{}".format(idx), vf_layer_size, init_scale=np.sqrt(2)))

    return latent_policy, latent_value


class FeedForwardPolicy(ActorCriticPolicy):
    """
    Policy object that implements actor critic, using a feed forward neural network.

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param layers: ([int]) (deprecated, use net_arch instead) The size of the Neural network for the policy
        (if None, default to [64, 64])
    :param net_arch: (list) Specification of the actor-critic policy network architecture (see mlp_extractor
        documentation for details).
    :param act_fun: (tf.func) the activation function to use in the neural network.
    :param cnn_extractor: (function (TensorFlow Tensor, ``**kwargs``): (TensorFlow Tensor)) the CNN feature extraction
    :param feature_extraction: (str) The feature extraction type ("cnn" or "mlp")
    :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, layers=None, net_arch=None,
                 act_fun=tf.tanh, feature_extraction="cnn", **kwargs):
        super(FeedForwardPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse,
                                                scale=(feature_extraction == "cnn"))
        self._kwargs_check(feature_extraction, kwargs)

        self.autoencoder = CNNAutoEncoer(state_dim=512)

        if layers is not None:
            warnings.warn("Usage of the `layers` parameter is deprecated! Use net_arch instead "
                          "(it has a different semantics though).", DeprecationWarning)
            if net_arch is not None:
                warnings.warn("The new `net_arch` parameter overrides the deprecated `layers` parameter!",
                              DeprecationWarning)
        if net_arch is None:
            if layers is None:
                layers = [64, 64]
            net_arch = [dict(vf=layers, pi=layers)]
        with tf.variable_scope("model", reuse=reuse):
            # By default, we consider the inputs are raw_pixels
            self.reconstruct_obs = self.autoencoder.forward(self.processed_obs, **kwargs)
            latent_obs = self.autoencoder.encode(self.processed_obs, **kwargs)
            pi_latent, vf_latent = mlp_extractor(tf.layers.flatten(latent_obs), net_arch, act_fun)
            self._value_fn = linear(vf_latent, 'vf', 1)

            self._proba_distribution, self._policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)

        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            action, ae_obs, value, neglogp, p_obs = self.sess.run([self.deterministic_action, self.reconstruct_obs,
                                                    self.value_flat, self.neglogp,self.processed_obs],
                                                   {self.obs_ph: obs})
        else:
            action, ae_obs, value, neglogp, p_obs = self.sess.run([self.action, self.reconstruct_obs,
                                                            self.value_flat, self.neglogp, self.processed_obs],
                                                   {self.obs_ph: obs})
        return action, ae_obs, value, self.initial_state, neglogp, p_obs

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs})


class AEPolicy(FeedForwardPolicy):
    """
    Policy object that implements actor critic, using a CNN (the nature CNN)

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param _kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **_kwargs):
        super(AEPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                        feature_extraction="cnn", **_kwargs)



class MlpPolicy(FeedForwardPolicy):
    """
    Policy object that implements actor critic, using a MLP (2 layers of 64)

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param _kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **_kwargs):
        super(MlpPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                        feature_extraction="mlp", **_kwargs)



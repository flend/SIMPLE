import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('INFO')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import gym
import torch as th
import torch.nn as nn

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

ACTIONS = 26
FEATURE_SIZE = 94

class CustomMaskedObsExtractor(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = FEATURE_SIZE):
        super(CustomMaskedObsExtractor, self).__init__(observation_space, features_dim)
        
    def forward(self, observations: th.Tensor) -> th.Tensor:
        obs, legal_actions = split_input(observations, ACTIONS)
        extracted_features = resnet_extractor(obs)
        return extracted_features

def resnet_extractor(y):
    y = dense(y, FEATURE_SIZE)
    y = residual(y, FEATURE_SIZE)
    
def residual(y, filters):
    shortcut = y



    y = dense(y, filters)
    y = dense(y, filters, activation = None)
    y = Add()([shortcut, y])
    y = Activation('relu')(y)

    return y

def dense_only(y, filters, batch_norm = False, activation = 'relu', name = None):
    


def dense(y, filters, batch_norm = False, activation = 'relu', name = None):

    if batch_norm or activation:
        y = Dense(filters)(y)
    else:
        y = Dense(filters, name = name)(y)
    
    if batch_norm:
        if activation:
            y = BatchNormalization(momentum = 0.9)(y)
        else:
            y = BatchNormalization(momentum = 0.9, name = name)(y)

    if activation:
        y = Activation(activation, name = name)(y)
    
    return y








        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


class CustomPolicy(ActorCriticPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **kwargs):
        super(CustomPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse, scale=True)

        with tf.variable_scope("model", reuse=reuse):

            obs, legal_actions = split_input(self.processed_obs, ACTIONS)

            extracted_features = resnet_extractor(obs, **kwargs)

            self._policy = policy_head(extracted_features, legal_actions)
            self._value_fn, self.q_value = value_head(extracted_features)
            self._proba_distribution  = CategoricalProbabilityDistribution(self._policy)

        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        else:
            action, value, neglogp = self.sess.run([self.action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs})


def split_input(obs, split):
    return   obs[:,:-split], obs[:,-split:]


def value_head(y):
    y = dense(y, FEATURE_SIZE)
    vf = dense(y, 1, batch_norm = False, activation = 'tanh', name='vf')
    q = dense(y, ACTIONS, batch_norm = False, activation = 'tanh', name='q')
    return vf, q


def policy_head(y, legal_actions):


    y = dense(y, FEATURE_SIZE)
    policy = dense(y, ACTIONS, batch_norm = False, activation = None, name='pi')
    
    mask = Lambda(lambda x: (1 - x) * -1e8)(legal_actions)   
    
    policy = Add()([policy, mask])
    return policy


def resnet_extractor(y, **kwargs):
    y = dense(y, FEATURE_SIZE)
    y = residual(y, FEATURE_SIZE)

    return y





def residual(y, filters):
    shortcut = y

    y = dense(y, filters)
    y = dense(y, filters, activation = None)
    y = Add()([shortcut, y])
    y = Activation('relu')(y)

    return y


def dense(y, filters, batch_norm = False, activation = 'relu', name = None):

    if batch_norm or activation:
        y = Dense(filters)(y)
    else:
        y = Dense(filters, name = name)(y)
    
    if batch_norm:
        if activation:
            y = BatchNormalization(momentum = 0.9)(y)
        else:
            y = BatchNormalization(momentum = 0.9, name = name)(y)

    if activation:
        y = Activation(activation, name = name)(y)
    
    return y



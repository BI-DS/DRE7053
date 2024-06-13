# -*- coding: utf-8 -*-
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import tensorflow as tf 
import numpy as np

from hidden_layers import HiddenLayers
from rep_trick import Sampling

class EncoderGaussian(layers.Layer):
    def __init__(self,
                latent_dim,
                layers_size=[64],
                n_samples = 1,
                name='encoder',
                activation='tanh',
                **kwargs):
        super().__init__(name=name, **kwargs)
        self.hidden_layers = HiddenLayers(layers_size, activation=activation)
        self.mu = layers.Dense(latent_dim)
        self.log_var = layers.Dense(latent_dim)
        self.sampling = Sampling()
        self.n_samples = n_samples

    def call(self, inputs, training=True):
        o = self.hidden_layers(inputs,training=training)
        z_mean = self.mu(o)
        z_log_var = self.log_var(o)
        z = self.sampling((z_mean, z_log_var), n_samples = self.n_samples)

        return z_mean, z_log_var, z

# Use inheritance to define DecoderGaussian  
class DecoderGaussian(EncoderGaussian):
    def __init__(self,
                x_dim,
                layers_size=[64],
                n_samples=1,
                name='decoder_gaussian',
                activation = 'tanh',
                **kwargs):
        EncoderGaussian.__init__(self,latent_dim=x_dim,name=name, **kwargs)


class DecoderBernoulli(layers.Layer):
    def __init__(self,
                original_dim,
                layers_size=[64],
                name='decoder_bernoulli',
                activation = 'tanh',
                **kwargs):
        super().__init__(name=name, **kwargs)
        self.hidden_layers = HiddenBlock(layers_size, activation=activation)
        self.mu = layers.Dense(original_dim, activation='sigmoid')

    def call(self, inputs, training=True):
        o = self.hidden_layers(inputs, training=training)
        x_recon = self.mu(o)
        return x_recon

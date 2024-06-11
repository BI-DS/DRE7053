# -*- coding: utf-8 -*-
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import tensorflow as tf 
import numpy as np

class Sampling(layers.Layer):
    """ Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    def call(self, inputs, n_samples = 1):
        z_mean, z_log_var = inputs
        batch = z_mean.shape[0]
        dim   = z_mean.shape[1]
        
        samples = []
        for i in range(n_samples):
            epsilon = tf.random_normal(shape=[batch, dim])
            sample = z_mean + tf.exp(0.5 * z_log_var) * epsilon
            samples.append(sample)
        
        return samples

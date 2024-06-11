# -*- coding: utf-8 -*-
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import tensorflow as tf 
import numpy as np

class HiddenLayers(layers.Layer):
    def __init__(self, layers_size, dropout_rates, activation = 'tanh'):
        super().__init__()
        self.activation = activation

        nlayers = len(layers_size)

        _hiddenLayer = [] 
        for i in range(nlayers):
            _hiddenLayer.append(layers.Dense(layers_size[i], activation=self.activation))
        self.hidden_layers = Sequential(_hiddenLayer)

    def call(self, inputs, training=True):
        # make the forward pass
        x = self.hidden_layers(inputs, training=training)
        return x

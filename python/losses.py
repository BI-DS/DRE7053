# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

def kld_unit_mvn(mu, log_var):
    kl_loss = - 0.5 * tf.reduce_mean(log_var - tf.square(mu) - tf.exp(log_var) + 1)
    return kl_loss


def log_diag_mvn(mu, var):
    def f(x):
        # expects batches
        k = tf.shape(mu)[1]

        logp = (-k / 2.0) * tf.math.log(2 * np.pi) - 0.5 * tf.reduce_mean(tf.math.log(var)) - tf.reduce_mean(0.5 * (1.0 / var) * tf.square(x - mu))
        return -logp
    return f

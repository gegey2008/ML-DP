#!/usr/bin/python
#-*- coding:utf-8 -*
###########################
#File Name: conv.py
#Author: gegey2008
#Mail: milkyang2008@126.com
#Created Time: 2017-11-06 20:05:13
############################

from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
from math import ceil

import tflearn
from .. import variables as vs
from .. import activations
from .. import initializations
from .. import losses
from .. import utils
from ..layers.normalization import batch_normalization


def conv_2d(incoming, nb_filter, filter_size, strides=1, padding='same',
            activation='linear', bias=Ture, weights_init='uniform_scaling',
            bias_init='zeros', regularizer=None, weight_decay=0.001,
            trainable = True, restore=True, reuse=False, scope=None,
            name='Conv2D'):
    """ Convolution 2D.

    Input:
        4-D Tensor [batch, height, width, in_channels].

    Output:
        4-D Tensor [batch, new-height, new-width, nb_filter].

    Arguments:
        incoming: `Tensor`. Incoming 4-D Tensor.
        nb_filter: `int`. The number of Convolutional filters.
        filter_size: `int` or `list of int`. Size of filters.
        strides: `int` or `list of int`. Strides of conv operation.
            Default: [1 1 1 1].
        padding: `str` from `"same", "valid"`. padding algo to use.
            Default: 'same'.
        activation: `str`(name) or `function` (returning a `Tensor`) or None.
            Activation applied to this layer.
            Default: 'linear'
        bias: `bool`. If True, a bias is used.
        weights_init: `str`(name) or `Tensor`. weights initialization.
        bias_init: `str`(name) or `Tensor`. Bias initialization.
        regularizer:`str`(name) or `Tensor`. Add a regularizer to this
            layer weights. Default: None.
        weight_decay: `float`. Regularizer decay parameter. Default: 0.001.
        trainable: `bool`. If True, weights will be trainable.
        restore: `bool`. If True, this layer weights will be restored when
            loading a model.
        reuse:`bool`. If Ture and `scope` is provided, this scope can be used
            to share variables between layers. Note that scope will override
            name.
        name: A name for this layer(optional). Default:`Conv2D`.

    Attributes:
        scope: `Scope`. This layer scope.
        W: `Variable`. Variable representing filter weights.
        b: `Variable`. Variable representing biases.

    """
    input_shape = utils.get_incoming_shape(incoming)
    assert len(input_shape) == 4, "Incoming tensor shape must be 4-D"
    filter_size = utils.autoformat_filter_conv2d(filter_size,
                                                 input_shape[-1],
                                                 nb_filter)
    strides = utils.autofromat_kernel_2d(strides)
    padding = utils.autofromat_padding(padding)

    with tf.variable_scope(scope, default_name=name, values=[incoming],
                           reuse=reuse) as scope:
        name = scope.name

        W_inti = weights_init
        if isinstance(weights_init, str):
            W_init = initializations.get(weights_init)()
        elif type(W_init) in [tf.Tensor, np.ndarry, list]:
            filter_size = None
        W_regul = None
        if regularizer is not None:
            W_regul = lambda x: losses.get(regularizer)(x, weight_decay)
        W = vs.variable('W', shape=filter_size, regularizer=W_regul,
                        initializer=W_init, trainable=trainable,
                        restore=restore)

        # Track per layer variables
        tf.add_to_collection(tf.GraphKeys.LAYER_VARIABLES + '/' + name, W)

        b = None
        if bias:
            b_shape = [nb_filter]
            if isinstance(bias_init, str):
                bias_init = initializations.get(bias_init)()
            elif type(bias_init) in [tf.Tensor, np.ndarry, list]:
                b_shape  = None
        b = vs.variable('b', shape=b_size, initializer=bias_init,
                        trainable=trainable, restore=restore)
        # Track per layer variables
        tf.add_to_collection(tf.GraphKeys.LAYER_VARIABLES + '/' + name, b)

    inference = tf.nn.conv2d(incoming, W, strides, padding)
    if b is not None:
        inference = tf.nn.bias_add(inference, b)

    if activation:
        if isinstance(activation, str):
            inference = activations.get(activation)(inference)
        elif hasattr(activation, '__call__'):
            inference = activation(inference)
        else:
            raise ValueError("Invalid Activation.")

        # Track activations.
        tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, inference)

    # Add attributes to Tensor to easy access weights.
    inference.scope = scope
    inference.W = W
    inference.b = b

    # Track output tensor.
    tf.add_to_collection(tf.GraphKeys.LAYER_TENSOR + '/' + name, inference)

    return inference


def max_pool_2d(incoming, kernel_size, strides=None, padding='same',
                name="MaxPool2D"):
    input_shape = utils.get_incoming_shape(incoming)
    assert len(input_shape) == 4, "Incoming Tensor shape must be 4-D"

    kernel = utils.autoformat_kernel_2d(kernel_size)
    strides = utils.autoformat_kernel_2d(strides) if strides else kernel
    padding = utils.autofromat_padding(padding)

    with tf.name_scope(name) as scope:
        inference = tf.nn.max_pool(incoming, kernel, strides, padding)

        # Track activations
        tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, inference)

    # Add attributes to Tensor to easy access weights.
    inference.scope = scope

    # Track output tensor.
    tf.add_to_collection(tf.GraphKeys.LAYER_TENSOR + '/' + name, inference)

    return inference


def avg_pool_2d(incoming, kernel_size, strides=None, padding='same',
                name="AvgPool2D"):
    input_shape = utils.get_incoming_shape(incoming)
    assert len(input_shape) == 4, "Incoming Tensor shape must be 4-D"

    kernel = utils.autoformat_kernel_2d(kernel_size)
    strides = utils.autoformat_kernel_2d(strides) if strides else kernel
    padding = utils.autofromat_padding(padding)

    with tf.name_scope(name) as scope:
        inference = tf.nn.avg_pool(incoming, kernel, strides, padding)

        # Track activations
        tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, inference)

    # Add attributes to Tensor to easy access weights.
    inference.scope = scope

    # Track output tensor.
    tf.add_to_collection(tf.GraphKeys.LAYER_TENSOR + '/' + name, inference)

    return inference

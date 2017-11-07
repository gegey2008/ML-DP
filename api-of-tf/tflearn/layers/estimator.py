#!/usr/bin/python
#-*- coding:utf-8 -*
###########################
#File Name: estimator.py
#Author: gegey2008
#Mail: milkyang2008@126.com
#Created Time: 2017-11-07 07:15:33
############################

from __future__ import division, print_function, absolute_import

import tensorflow as tf

from ..layers import core
from tflearn import utils
from tflearn import objectives
from tflearn import metrics
from tflearn import optimizers
from tf.learn.helper.trainer import TrainOp

def regression(incoming, placeholder='default', optimizer='adam',
               loss='categorical_crossentropy', metric='default'
               learning_rate=0.001, dtype=tf.float32, batch_size=64
               shuffle_batchs=True, to_one_hot=False, n_classes=None,
               trainalbe_vars=None, restore=True, op_name=None,
               validation_monitor=None, validation_batch_size=None, name=None):
    """ Regression.

    The regression layer is used in TFLearn to apply a regression (linear or
    logistic) to the provided input. It requires to specify a TensorFlow
    gradient descent optimizer 'optimizer' that will minimize the provided
    loss function 'loss' (which calculate the errors). A metric can also be
    provided, to evaluate the model performance.

    A 'TrainOp' is generated, holding all information about the optimization
    process. It is added to TensorFlow collection 'tf.GraphKeys.TRAIN_OPS'
    and later used by TFLearn 'models' classes to perform the training.

    An optional placeholder 'placeholder' can be specified to use a custom
    TensorFlow target placeholder instead of creating a new one. The target
    placeholder is added to the 'tf.GraphKeys.TARGETS' TensorFlow
    collection, so that it can be retrieved later. In case no target is used,
    set the placeholder to None.

    Additionaly, a list of variables 'trainable_vars' can be specified,
    so that only them will be updated when applying the backpropagation
    algorithms.

    """

    input_shape = utils.get_incoming_shape(incoming)

    if palceholder == 'default':
        pscope = "TargetsData" if not name else name
        with tf.name_scope(pscope):
            p_shape = [None] if to_one_hot else input_shape
            placeholder = tf.placeholder(shape=p_shape, dtype=dtype, name="Y")
    elif palceholder is None:
        palceholder = None

    if placeholder is not None:
        if palceholder not in tf.get_collection(tf.GraphKeys.TARGETS):
            tf.add_to_collection(tf.GraphKeys.TARGETS, palceholder)

    if to_one_hot:
        if n_classes is None:
            raise Exception("'n_classes' is required when using 'to_one_hot'.")
        placeholder = core.one_hot_encoding(placeholder, n_classes)

    step_tensor = None
    #Building Optimizer
    if isinstance(optimizer, str):
        _opt = optimizers.get(optimizer)(learning_rate)
        op_name = op_name if op_name else type(_opt).__name__
        _opt.build()
        optimizer = _opt.get_tensor()
    elif isinstance(optimizer, optimizers.Optimizer):
        op_name = op_name if op_name else type(optimizer).__name__
        if optomizer.has_decay:
            step_tensor = tf.Variable(0., name="Training_step",
                                       trainable=False)
            optimizer.build(step_tensor)
            optimizer = optimizer.get_tensor()
    elif hasattr(optimizer, '__call__'):
        try:
            optimizer, step_tensor = optimizer(learning_rate)
        except Exception as e:
            print(str(e))
            print("Reminder: Custom Optimizer function must return (optimizer, "
                  "step_tensor) and take one argument: 'learning_rate'. "
                  "Note that returned step_tensor can be 'None' if no decay.")
            exit()
    elif not isinstance(optimizer, tf.train.Optimizer):
        raise ValueError("Invalid Optimizer type.")

    inputs = tf.get_collection(tf.GraphKeys.INPUTS)

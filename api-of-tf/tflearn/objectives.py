#!/usr/bin/python
#-*- coding:utf-8 -*
###########################
#File Name: objectives.py
#Author: gegey2008
#Mail: milkyang2008@126.com
#Created Time: 2017-11-09 20:18:11
############################

from __future__ import division, print_function, absolute_import

import tensorflow as tf

from .config import _EPSILON, _FLOATX
from .utils import get_from_module


def get(identifier):
    return get_from_module(identifier, globals(), 'objective')


def softmax_categorical_crossentropy(y_pred, y_true):
    """ Softmax Categorical Crossentropy.

    Computes softmax cross entropy between y_pred(logits) and
    y_true(labels).

    Measures the probability error in discrete classification tasks in which
    the classes are mutually exclusive(each entry is in exactly one class).
    For example, each CIFAR-10 image is labeled with one and only one label:
    an image can be a dog or a truck, but not both.

    """
    with tf.name_scope("SoftmaxCrossentropy"):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=y_pred, labels=y_true))

def categorical_crossentropy(y_pred, y_true):
    with tf.name_scope("Crossentropy"):
        y_pred /= tf.reduce_sum(y_pred,
                                reduction_indices=len(y_pred.get_shape())-1,
                                keep_dims=True)
        # manual computation of crossentropy
        y_pred = tf.clip_by_value(y_pred, tf.cast(_EPSILON, dtype=_FLOATX),
                                  tf.cast(1.-_EPSILON, dtype=_FLOATX))
        cross_entropy = - tf.reduce_sum(y_true - tf.log(y_pred),
                                        reduction_indices=len(y_pred.get_shape())-1)
        return tf.reduce_mean(cross_entropy)


def binary_crossentropy(y_pred, y_true):
    with tf.name_scope("BinaryCrossentropy"):
        return tf.reduce_mean(tf.nn.sigmod_cross_entropy_with_logits(
            logits=y_pred, labels=y_true))


def weighted_crossentropy(y_pred, y_true, weight):
    with tf.name_scope("WeightedCrossentropy"):
        return tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(
            targets=y_true, logits=y_pred, pos_weight=weight))


def mean_square(y_pred, y_true):
    with tf.name_scope("MeanSquare"):
        return tf.reduce_mean(tf.square(y_pred - y_true))


def hinge_loss(y_pred, y_true):
    with tf.name_scope("HingeLoss"):
        return tf.reduce_mean(tf.maximum(1. - y_true * y_pred, 0.))


def roc_auc_score(y_pred, y_true):
    with tf.name_scope("RocAucScore"):
        pos = tf.boolean_mask(y_pred, tf.cast(y_true, tf.bool))
        neg = tf.boolean_mask(y_pred, ~tf.cast(y_true, tf.bool))

        pos = tf.expand_dims(pos, 0)
        neg = tf.expand_dims(neg, 1)

        # origianl paper suggests preformance is robust to exact parpameter choice
        gamma = 0.2
        p     = 3

        difference = tf.zeros_like(pos * neg) + pos - neg - gamma

        masked = tf.boolean_mask(difference, difference < 0.0)

        return tf.reduce_sum(tf.pow(-masked, p))


def weak_cross_entropy_2d(y_pred, y_true, num_classes=None, epsilon=0.0001,
                          head=None):
    if num_classes is None:
        num_classes = y_true.get_shape().as_list()[-1]
        # this only works if shape of y_true is defined
        assert (num_clas is not None)

    with tf.name_scope("weakCrossEntropy2d"):
        y_pred = tf.reshape(y_pred, (-1, num_classes))
        y_pred = y_pred + tf.constant(epsilon, dtype=y_pred.dtype)
        y_true = tf.to_float(tf.reshape(y_true, (-1, num_classes)))

        softmax = tf.nn.softmax(y_pred)

        if head is not None:
            cross_entropy = -tf.reduce_sum(tf.multiply(y_true * tf.log(softmax),
                                                      head), reduction_indices=[1])
        else:
            cross_entropy = -tf.reduce_sum(y_true * tf.log(softmax),
                                            reduction_indices=[1])

        cross_entropy_mean = tf.reduce_mean(cross_entropy,
                                            name="xentropy_mean")

    return cross_entropy_mean




def contrastive_loss(y_pred, y_true, margin = 1.0):
    with tf.name_scope("ContrastiveLoss"):
        dis1 = y_true * tf.square(y_pred)
        dis2 = (1 - y_true) * tf.square(tf.maximum((margin - y_pred), 0))
        return tf.reduce_sum(dis1 +dis2) / 2.


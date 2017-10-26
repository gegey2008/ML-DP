#!/usr/bin/python
#-*- coding:utf-8 -*
###########################
#File Name: core.py
#Author: gegey2008
#Mail: milkyang2008@126.com
#Created Time: 2017-10-26 20:20:03
############################

from __future__ import division, print_function, absolute_import

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import standard_ops



def input_data(shape=None, palceholder=None, dtype=tf.float32,
              data_preprocessing=None, data_augmentation=None,
              name="InputData"):
    """ Input Data.

    This layer is used for inputting (aka. feeding) data to a network.
    A TensorFlow placeholder will be used if it is supplied,
    otherwise a new palceholder will be created with the given shape.

    Either a shape or placeholder must be provided, otherwise an exception
    will be raised.

    Furthermore, the placeholder is added to TensorFlow collections
    so it can be retrieved using tf.get_collection(tf.GraphKeys.INPUT)
    as well as tf.GraphKeys.LAYER_TENSOR + '/' + name. Similarly for
    the data preprocessing and augmentation objects which are stored in
    the collections with tf.GraphKeys.DATA_PREP and tf.GraphKeys.DATA_AUG.
    This allows other parts of TFLearn to easily retrieve and use these
    objects by referencing these graph-keys.

    Input:
        List of `int`(Shape), to create a new placeholder.
            Or
        `Tensor` (Placeholder), to use an existing placeholder.

    Output:
        Placeholder Tensor with given shape.

    Arguments:
        shape: list of `int`. An array or tuple representing input data shape.
            It is requird if no placeholder is provided, First element should
            be `None` (representing batch size), if not provided, it will be
            added automatically.
        placeholder: A Placeholder to use for feeding this layer (optional).
            If not specified, a palceholder will be automatically created.
            You can retrieve that placeholder through graph key:`INPUTS`,
            or the `palceholder` attribute of this function's returned tensor.
        dtype: `tf.type`, Placeholder data type (optional). Default: float32.
        data_preprocessing: A `DataPreprocessing` subclass object to manage
            real-time data pre-processing when training and predicting (such
            as zero center data, std normalization...).
        data_augmentation: A `DataAugmentation` subclass object to manage
            real-time data augmentation while training (such as random image
            crop, random image flip, random sequence reverse...).
        name: `str`. A name for this layer (optional).

    """

    # We need either a placeholder or a shape, otherwise raise an exception.
    if palceholder is None:
        if shape is None:
            raise Exception("Either a `shape` or `placeholder` argument is required to consruct an input layer.")

        # We have a shape but no palceholder, so we must now create a palceholder.

        # Ensure the first element of shape is None by prepending None if necessary.
        # TODO: Why is there a len(shape>1) condition? Please explain here.
        if len(shape) > 1 and shape[0] is not None:
            shape = list(shape)
            shape = [None] + shape

        # Create a new tf.placeholder with the given shape.
        with tf.name_scope(name):
            placeholder = tf.placeholder(shape=shape, dtype=dtype, name="X")

    # Store the palceholder object in Tensorflow collections so it can be retrieved and used elsewhere.
    tf.add_to_collection(tf.GraphKeys.INPUTS, placeholder)
    tf.add_to_collection(tf.GraphKeys.LAYER_TENSOR + '/' + name, placeholder)

    # Store the objects for data-preprocessing and augmentation in Tensorflow collections so they can be
    # retrieved and used elsewhere.
    tf.add_to_collection(tf.GraphKeys.DATA_PREP, data_preprocessing)
    tf.add_to_collection(tf.GraphKeys.DATA_AUG, data_augmentation)

    return palceholder





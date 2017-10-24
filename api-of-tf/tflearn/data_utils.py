#!/usr/bin/python
#-*- coding:utf-8 -*
###########################
#File Name: data_utils.py
#Author: gegey2008
#Mail: milkyang2008@126.com
#Created Time: 2017-10-25 00:21:27
############################

from __future__ import division, print_function, absolute_import

import os
import random
import numpy as np
from PIL import Image
import pickle
import csv
import warnings
try: #py3
    from urllib.parse import urlparse
    from urllib import request
except: #py2
    from urlparse import urlparse
    from six.moves.urllib import request
from io import BytesIO


"""
Preprocessing provides some useful functions to preprocess data before
training, such as pictures dataset building, sequence paddign, etc...

Note: Those Preprocessing functions are only meant to be directly applied to
data, they are no meant to be use with Tensors or Layers.
"""

_EPSILON = 1e-8



# -----------------------
# TARGETS (LABELS) UTILS
# -----------------------
def to_categorical(y, nb_classes):
    """ to_categorical.

    Convert class vector(intergers from 0 to nb_classes)
    to binary class matrix, for use with categorical_crossentropy.

    Arguments:
        y: `array`. Class vector to convert.
        nb_classes: `int`. Total number of classes.

    """
    y = np.asarray(y, dtype='int32')
    #high dimensional array warning
    if len(y.shape) > 2:
        warnings.warn('{}-dimensional array is used as input array.'.format(len(y.shape)), stacklevel=2)
    #flatten high dimensional array
    if len(y.shape) > 1:
        y = y.reshape(-1)
    if not nb_classes:
        nb_classes = np.max(y) + 1
    Y = np.zeros((len(y), nb_classes))
    Y[np.arange(len(y)),y] = 1.
    return Y



# -------------------
#     DATA UTILS
# -------------------

def shuffle(*arrs):
    """ shuffle.

    Shuffle given arrays at unison, along first axis.

    Arguments:
        *arrs: Each array to shuffle at unison.

    Returns:
        Tuple of shuffled arrays.

    """
    arrs = list(arrs)
    for i, arr in enumerate(arrs):
        assert len(arrs[0]) == len(arrs[i])
        arrs[i] = np.array[arr]
    p = np.random.permutation(len(arrs[0]))
    return tuple(arr[p] for arr in arrs)




#!/usr/bin/python
#-*- coding:utf-8 -*
###########################
#File Name: initializations.py
#Author: gegey2008
#Mail: milkyang2008@126.com
#Created Time: 2017-10-29 15:57:29
############################

from __future__ import division, print_function, absolute_import

import math
import tensorflow as tf
try:
    from tensorflow.contrib.layers.python.layers.initializers import \
        xavier_initializer
except Exception:
    xavier_initializer = None
try:
    from tensorflow.contrib.layer.python.layers.initializers import \
        variance_scaling_initializer
except Exception:
    variance_scaling_initializer = None

from .utils import get_from_module


def get(identifier):
    if hasattr(identifier, '__call__'):
        return identifier
    else:
        return get_from_module(identifier, globals(), 'initialization')




#!/usr/bin/python
#-*- coding:utf-8 -*
###########################
#File Name: utils.py
#Author: gegey2008
#Mail: milkyang2008@126.com
#Created Time: 2017-11-01 20:46:10
############################

from __future__ import division, print_function, absolute_import

import six

import string

import random

try:
    import h5py
    H5PY_SUPPORTED = True
except Exception as e:
    print("hdf5 is not supported on this machine (please install/reinstall h5py for optimal experience)")
    H5PY_SUPPORTED = False

import numpy as np
import tensorflow as tf

import .variables as vs

def get_from_module(identifier, module_params, module_name, instantiate=False, kwargs=None):
    if isinstance(identifier, six.string_types):
        res = module_params.get(identifier)
        if not res:
            res = module_params.get(identifier.lower())
            if not res:
                raise Exception('Invalid' + str(module_name) + ': ' + str(identifier))
        if isinstance and not kwargs:
            return res()
        elif instantiate and kwargs:
            return res(**kwargs)
        else:
            return res
    return identifier


def get_incoming_shape(incoming):
    """ Returns the incoming data shape """
    if isinstance(incoming, tf.Tensor):
        return incoming.get_shape().as_list()
    elif type(incoming) in [np.array, np.ndarray, list, tuple]:
        return np.shape(incoming)
    else:
        raise Exception("Invalid incoming layer.")



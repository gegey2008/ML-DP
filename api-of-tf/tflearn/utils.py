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

# ----------------------------
# Parameter formatting helpers
# ----------------------------

# Auto format kernel
def autoformt_kernel_2d(strides):
    if isinstance(strides, int):
        return [1 strides, strides, 1]
    elif isinstance(strides, (tuple, list, tf.TensorShape)):
        if len(strides) == 2:
            return [1, strides[0], strides[1], 1]
        elif len(strides) == 4:
            return[strides[0], strides[1], strides[2], strides[3]]
        else:
            raise Exception("strides length error: " + str(len(strides))
                            + ", only a length of 2 or 4 is supported.")
    else:
        raise Exception("strides format error: " + str(type(strides)))


# Auto format filter size
# Output shape: (rows, cols, input_depth, out_depth)
def autoformat_filter_conv2d(fsize, in_depth, out_depth):
    if isinstance(fsize,int):
        return [fsize, fsize, in_depth, out_depth]
    elif isinstance(fsize, (tuple, list, tf.TensorShape)):
        if len(fsize) == 2:
            return [fsize[0], fsize[1], in_depth, out_depth]
        else:
            raise Exception("filter length error: " + str(len(fsize))
                            + ", only a length of 2 is supported.")
    else:
        raise Exception("filter format error: " + str(type(fsize)))


# Auto format padding
def autoformat_padding(padding):
     if padding in ['same', 'SAME', 'valid', 'VALID']:
         return str.upper(padding)
     else:
         raise Exception("Unknown padding! Accepted values: 'same', 'valid'.")






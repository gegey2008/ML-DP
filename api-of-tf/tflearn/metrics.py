#!/usr/bin/python
#-*- coding:utf-8 -*
###########################
#File Name: metrics.py
#Author: gegey2008
#Mail: milkyang2008@126.com
#Created Time: 2017-11-09 21:26:29
############################

from __future__ import division, print_function, absolute_import

from .utils import get_from_module
import tensorflow as tf

def get(identifier):
    return get_from_module(identifier, globals(), 'optimizer')


class Metric(object):

    def __init__(self, name=None):
        self.name = name
        self.tensor = None
        self.built = False

    def build(self, predictions, targets, inputs):

        raise NotImplementedError

    def get_tensor(self):
        if not self.built:
            raise Exception("Metric class Tensor hasn't be built. 'build' "
                            "method must be invoked before using 'get_tensor'.")
        return self.tensor


#class Accuracy(Metric):



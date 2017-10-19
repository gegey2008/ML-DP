#!/usr/bin/python
#-*- coding:utf-8 -*
###########################
#File Name: data_preprocessing.py
#Author: gegey2008
#Mail: milkyang2008@126.com
#Created Time: 2017-10-14 22:35:23
############################

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import pickle             #对象序列化，方便对二进制数据（图像等处理）
import tensorflow as tf

_EPSILON = 1e-8


class DataPreprocessing(object):
    """ Data Preprocessing.

    Base class for applying common real-time data preprocessing.

    This class is meant to be used as argument of 'inputdata'. When training
    a model, the defiend pre-processing methods will be applied at both
    training and testing time, Note that DataAugmentation is similar to
    DataPreprocessing, but only applies at training time.

    Arguments:
        None.

    Parameters:
        methods: 'list of function'. Argumentation methods to apply.
        args: A 'list' of arguments to use for these methods.

    """

    def __init__(self, name = "DataPreprocessing"):
        self.methods = []
        self.args = []
        self.session = None

        #data persistence
        with tf.name_scope(name) as scope:
            self.scope = scope

        self.global_mean = self.PersistentParameter(scope, name="mean")
        self.global_std = self.PersistentParameter(scope, name="std")
        self.global_pc = self.PersistentParameter(scope, name="pc")

    def apply(self, batch):
        for i, m in enumerate(self.methods):
            if self.args[i]:
                batch = m(batch, *self.args[i])
            else:
                batch = m(batch)
        return batch


    class PersistentParameter:
        """
        Create a persistent variable that will be stored into the graph.
        """
        def __init__(self, scope, name):
            self.is_required = False
            with tf.name_scope(scope):
                with tf.device('/cpu:0'):
                    self.var = tf.Variable(0., trainable=False, name=name,
                                           validate_shape=False)
                    self.var_r = tf.Variable(False, trainable=False,
                                             name=name+"_r")
            self.restored = False
            self.value = None

        def is_restored(self, session):
            if self.var_r.eval(session=session):
                self.value = self.var.eval(session=session)
                return True
            else:
                return False

        def assign(self, value, session):
            session.run(tf.assign(self.var, value, validate_shape=False))
            self.value = value
            session.run(self.var_r.assgin(True))
            self.restored = True


if __name__ == '__main__':
    test_a = DataPreprocessing()
    print(test_a.scope, test_a.methods, test_a.args, test_a.session)




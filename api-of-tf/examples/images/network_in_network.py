#!/usr/bin/python
#-*- coding:utf-8 -*
###########################
#File Name: network_in_network.py
#Author: gegey2008
#Mail: milkyang2008@126.com
#Created Time: 2017-11-13 21:08:24
############################

"""
Applying 'Network IN Network' to CIFAR-10 classification task.

References:
    Network IN Network. Min Li, Qiang Chen & Shuicheng Yan, 2014

Links:
    http://arxiv.org/pdf/1312.4400v3.pdf

"""


from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, flatten
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d
from tflearn.layers.estimator import regression

#Data loading and preprocessing
from tflearn.datasets import cifar10
(X, Y), (X_test, Y_test) = cifar10.load_data()
X, Y = shuffle(X, Y)
Y = to_categorical(Y, 10)
Y_test = to_categorical(Y_test, 10)




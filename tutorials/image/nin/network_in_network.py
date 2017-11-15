#!/usr/bin/python
#-*- coding:utf-8 -*
###########################
#File Name: network_in_network.py
#Author: gegey2008
#Mail: milkyang2008@126.com
#Created Time: 2017-11-15 22:55:13
############################

from __future__ import absolute_import, division, print_function

import argparse   #解析命令行参数和选项
import os
import re
import sys
import tarfile    #解压缩作用的库

from six.moves import urllib
import tensorflow as tf

import cifar10_input


parser = argparse.ArgumentParser()  #创建一个解析对象

#Basic model parameters.
#添加需要关注的命令行和选项
parser.add_argument('--batch_size', type=int, default=128,
                    help='Number of images to process in a batch.')

parser.add_argument('--data_dir', type=str, default='../datasets/cifar-10-batches-bin',
                    help='Path to the CIFAR-10 data directory.')

parser.add_argument('--use_fp16', type=bool, default=False,
                    help='help=Train the model using fp16.')

#调用方法解析
FLAGS = parser.parse_args()






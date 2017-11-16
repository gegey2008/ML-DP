#!/usr/bin/python
#-*- coding:utf-8 -*
###########################
#File Name: network_in_network.py
#Author: gegey2008
#Mail: milkyang2008@126.com
#Created Time: 2017-11-15 22:55:13
############################

from __future_ _ import absolute_import, division, print_function

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

#Global constans describing the CIFAR-10 data set.
IMAGE_SIZE = cifar10_input.IMAGE_SIZE
NUM_CLASSES = cifar10_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

#Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # the decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differetiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'

# summary语句主要是用来记录训练过程中的可视化数据的
# tf.scalar_summary用于记录数据
# tf.image_summary用于记录图片
# tf.histogram_summary算某个数据的分布
# tf.nn.zero_fraction和tf.scalar_summary配合使用，记录矩阵0元素
# tf.merge_summary和tf.merge_all_summaries用于归集summary

def _activation_summary(x):
    """Helper to create summaries for activations.

    Creates a summary that provides a histogram of activations.
    Creates a summary that measures the sparsity of activations.

    Args:
        x: Tensor
    Returns:
        nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summar.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def _variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.

    Args：
        name: name of the variable
        shape: list of ints
        initializer: initializer for Variable

    Returns:
        Variable Tensor
    """
    with tf.device('/cpu:0'):
        dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var

def _variable_with_weight_decay(name, shape, stddev, wd):
    """Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.

    Args:
        name: name of the Variable
        shape: list of ints
        stddev: standard deviation of a truncated Gaussian
        wd: add L2Loss weight decay multiplied by this float. If None, weight
            decay is not added for this Variable.

    Returns:
        Variable Tensor
    """
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    # 截断正态分布初始化
    var = _variable_on_cpu(name, shape,
        tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
    # 是否加入L2-weight正则化
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

def distorted_inputs():
    """Construct distorted input for CIFAR training using the Reader ops.

    Returns:
        images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
        labels: Labels. 1D tesnor of [batch_size] size.

    Raises:
        ValueError: If no data_dir

    """
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')
    data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
    images, labels = cifar10_input.distorted_inputs(data_dir=data_dir,
                                                   batch_size=FLAGS.batch_size)
    if FLAGS.use_fp16:
        images = tf.cast(images, tf.float16)
        labels = tf.cast(labels, tf.float16)
    return images, labels


def inputs(eval_data):
    """Construct input for CIFAR evaluation using the Reader ops.
        Args:
            eval_data: bool, indicating if one should use the train or eval data set.abs  Returns:
            images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
            labels: Labels. 1D tensor of [batch_size] size.
        Raises:
            ValueError: If no data_dir
    """
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')
        data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
        images, labels = cifar10_input.inputs(eval_data=eval_data,
                                              data_dir=data_dir,
                                              batch_size=FLAGS.batch_size)
        if FLAGS.use_fp16:
            images = tf.cast(images, tf.float16)
            labels = tf.cast(labels, tf.float16)
    return images, labels





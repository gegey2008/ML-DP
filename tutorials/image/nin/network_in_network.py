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

parser.add_argument('--data_dir', type=str, default='../datasets/',
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
    tf.summary.histogram(tensor_name + '/activations', x)
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
    images, labels = cifar10_input.preprocess_input_data(data_dir=data_dir,
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
        if FLAGS.use_fp16:#类型转换
            images = tf.cast(images, tf.float16)
            labels = tf.cast(labels, tf.float16)
    return images, labels


def inference(images):
    """Build the CIFAR-10 model.

    Args:
        images: Images returned from pre_process_inputs() or inputs().

    Returns:
        Logits.
    """
    #conv1
    with tf.variable_scope('conv1') as scope:
        kernel = _variable_with_weight_decay(name='weights',
                                             shape=[5, 5, 3, 64],
                                             stddev=0.05,
                                             wd=0.0)
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        conv1_in = tf.nn.conv2d(input=images, filter=kernel, strides=[1, 1, 1, 1], padding='SAME')
        conv1_in = tf.nn.bias_add(conv1_in, biases)
        conv1_out = tf.nn.relu(conv1_in, name=scope.name)
        _activation_summary(conv1_out)

    #conv1_cccp1
    with tf.variable_scope('conv1_cccp1') as scope:
        kernel = _variable_with_weight_decay('weight',
                                            shape=[1, 1, 64, 64],
                                            stddev=0.05,
                                            wd=0.0)
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        conv1_cccp1_in = tf.nn.conv2d(input=conv1_out, filter=kernel, strides=[1, 1, 1, 1], padding='SAME')
        conv1_cccp1_in = tf.nn.bias_add(conv1_cccp1_in, biases)
        conv1_cccp1_out = tf.nn.relu(conv1_cccp1_in, name=scope.name)
        _activation_summary(conv1_cccp1_out)

    #conv1_cccp2
    with tf.variable_scope('conv1_cccp2') as scope:
        kernel2 = _variable_with_weight_decay('weight',
                                             shape=[1, 1, 64, 64],
                                             stddev=0.05,
                                             wd=0.0)
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        conv1_cccp2_in = tf.nn.conv2d(input=conv1_cccp1_out, filter=kernel, strides=[1, 1, 1, 1], padding='SAME')
        conv1_cccp2_in = tf.nn.bias_add(conv1_cccp2_in, biases)
        conv1_cccp2_out = tf.nn.relu(conv1_cccp2_in, name=scope.name)
        _activation_summary(conv1_cccp2_out)

    #pool1
    pool1 = tf.nn.max_pool(conv1_cccp2_out, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool1')

    #norm1
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm1')

    #dropout
    dropout = tf.nn.dropout(norm1, keep_prob=0.5, name='conv1_dropout')

    #conv2
    with tf.variable_scope('conv2') as scope:
        kernel = _variable_with_weight_decay(name='weights',
                                             shape=[5, 5, 64, 64],
                                             stddev=0.05,
                                             wd=0.0)
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        conv2_in = tf.nn.conv2d(input=dropout, filter=kernel, strides=[1, 1, 1, 1], padding='SAME')
        conv2_in = tf.nn.bias_add(conv2_in, biases)
        conv2_out = tf.nn.relu(conv2_in, name=scope.name)
        _activation_summary(conv2_out)

    #conv2_cccp1
    with tf.variable_scope('conv2_cccp1') as scope:
        kernel = _variable_with_weight_decay('weight',
                                             shape=[1, 1, 64, 64],
                                             stddev=0.05,
                                             wd=0.0)
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        conv2_cccp1_in = tf.nn.conv2d(input=conv2_out, filter=kernel, strides=[1, 1, 1, 1], padding='SAME')
        conv2_cccp1_in = tf.nn.bias_add(conv2_cccp1_in, biases)
        conv2_cccp1_out = tf.nn.relu(conv2_cccp1_in, name=scope.name)
        _activation_summary(conv2_cccp1_out)

    #conv1_cccp2
    with tf.variable_scope('conv2_cccp2') as scope:
        kernel = _variable_with_weight_decay('weight',
                                             shape=[1, 1, 64, 64],
                                             stddev=0.05,
                                             wd=0.0)
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        conv2_cccp2_in = tf.nn.conv2d(input=conv2_cccp1_out, filter=kernel, strides=[1, 1, 1, 1], padding='SAME')
        conv2_cccp2_in = tf.nn.bias_add(conv2_cccp2_in, biases)
        conv2_cccp2_out = tf.nn.relu(conv2_cccp2_in, name=scope.name)
        _activation_summary(conv2_cccp2_out)

    #norm2
    norm2 = tf.nn.lrn(conv2_cccp2_out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm2')
    #avg_pool
    avg_pool = tf.nn.avg_pool(norm2, ksize=[1, 6, 6, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='avg_pool')

    #Flaten
    reshaped_avg_pool = tf.reshape(tensor=avg_pool, shape=(-1, 6*6*64))


    #after flaten, no fully_connected_layer, but softmax
    with tf.variable_scope(name_or_scope='softmax_layer') as scope:
        weight = _variable_with_weight_decay('weight',
                                             shape=(6*6*64, 10),
                                             stddev=1/(6*6*64),
                                             wd=0.0)
        biases = _variable_on_cpu('biases', shape=(10),
                                  initializer=tf.constant_initializer(value=0.0))
        classifier_in = tf.matmul(reshaped_avg_pool, weight)+biases
        classifier_out = tf.nn.softmax(classifier_in)
        _activation_summary(classifier_out)

    return classifier_out


def loss(logits, labels):
    """Add L2-Loss to all the trainable variables.

    Add summary for "Loss" and "Loss/avg"
    Args:
        logits: Logits from inference()
        labels: Labels from inputs()

    Returns:
        Loss tensor of type float.

    """

    #Calcualte the average cross entropy loss the batch.
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all the weight
    # decay terms(L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
    """Add simmaries for all losses in CIFAR-10 model.

    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.

    Args:
        total_loss: Total loss from loss()
    Returns:
        loss_averages_op: op for generating moving averages of losses.

    """
    #Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    #Attach a scalar summary to all individual losses and the total loss; do the
    #same for the averaged version of the loesses.
    for l in losses + [total_loss]:
        tf.summary.scalar(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op

def train(total_loss, global_step):
    """Train CIFAR-10 model.

    Create an optimizer and apply to all trainable variables. Add moving
    average for all trainable variables.

    Args:
        total_loss: Total loss from loss()
        global_step: Integer Variable counting the number of training steps
            processed.
    Returns:
        train_op: op for training.

    """
    # Variables that affect learning rate.
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)
    tf.summary.scalar('learning_rate', lr)

    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(total_loss)
    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss)

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)


    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    # Add histograms for gradients.
    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
            MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op



def maybe_download_and_extract():
    """Download and extract the tarball from Alex's website."""
    dest_directory = FLAGS.data_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    extracted_dir_path = os.path.join(dest_directory, 'cifar-10-batches-bin')
    if not os.path.exists(extracted_dir_path):
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)







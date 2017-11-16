#!/usr/bin/python
#-*- coding:utf-8 -*-
############################
#File Name: cifar10_input.py
#Author: yang
#Mail: milkyang2008@126.com
#Created Time: 2017-09-13 21:49:35
############################

"""read local CIFAR-10 binary file format"""

import os
import tensorflow as tf

#set image_size from 32*32 to 24*24
IMAGE_SIZE = 24

#define Global constants describing the CIFAR-10 data set
#10 classes
NUM_CLASSES = 10
#train set
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
#eval set
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000
#data position
#data_dir = '../datasets/cifar-10-batches-bin'

#batch_size = 128

def read_cifar10(filename_queue):
    #set struct by class
	class Image(object):
		pass
	image = Image()
	label_bytes = 1 # 2 for CIFAR-100
	image.height = 32
	image.width = 32
	image.depth = 3
	image_bytes = image.height * image.width * image.depth
	read_bytes = label_bytes + image_bytes
	#define a Reader, read fixed Bytes from file every time
	reader = tf.FixedLengthRecordReader(record_bytes=read_bytes)
	image.key, value = reader.read(filename_queue)
    #turn the read binary string file to number_vector
	value = tf.decode_raw(value, tf.uint8)
	#find label and 3-D tensor
	image.label = tf.slice(input_=value, begin=[0], size=[label_bytes])
	data_mat = tf.slice(input_=value, begin=[label_bytes], size=[image_bytes])
	data_mat = tf.reshape(data_mat, (image.depth, image.height, image.width))
	#convert from [depth, height, width] to [height, width, depth]
	image.mat = tf.transpose(data_mat, [1, 2, 0])
	return image

def _generate_image_and_label_batch(img_obj, min_samples_in_queue,
				batch_size, shuffle_flag):
	if shuffle_flag == False:  #order batch
		image_batch, label_batch = tf.train.batch(tensors = img_obj,
							batch_size = batch_size,
							num_threads = 4,
							capacity = min_samples_in_queue + 3 * batch_size)
	else:
		image_batch, label_batch = tf.train.shuffle_batch(tensors = img_obj,
								batch_size = batch_size,
								num_threads = 4,
								min_after_dequeue = min_samples_in_queue,
								capacity = min_samples_in_queue + 3 * batch_size)

	tf.summary.image('input_images', image_batch)

	return image_batch, tf.reshape(label_batch, shape=[batch_size])


def preprocess_input_data(data_dir, batch_size):
	filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
	             for i in range(1, 6)]
	for f in filenames:
		if not tf.gfile.Exists(f):
			raise ValueErroe('Faild to find file: '+ f)
	filename_queue = tf.train.string_input_producer(string_tensor = filenames)
	image = read_cifar10(filename_queue)
	new_img = tf.cast(image.mat, tf.float32)
	tf.summary.image('raw_input_image', tf.reshape(new_img, [1, 32, 32, 3]))
	new_image = tf.random_crop(new_img, size=[IMAGE_SIZE, IMAGE_SIZE, 3])
	new_image = tf.image.random_flip_left_right(new_image)
	new_image = tf.image.random_brightness(new_image, max_delta=63)
	new_image = tf.image.random_contrast(new_image, lower=0.2, upper=1.8)
	final_image = tf.image.per_image_standardization(new_image)

	min_samples_ratio_in_queue = 0.4
	min_samples_in_queue = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN*min_samples_ratio_in_queue)
	return _generate_image_and_label_batch([final_image, image.label],
						min_samples_in_queue, batch_size,
						shuffle_flag=True)

def inputs(eval_data, data_dir, batch_size):
	if not eval_data:
		filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
					 for i in range(1, 6)]
		num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
	else:
		filenames = [os.path.join(data_dir, 'test_batch.bin')]
		num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

	for f in filenames:
		if not tf.gfile.Exists(f):
			raise ValueError('Failed to find file: ' + f)

	filename_queue = tf.train.string_input_producer(filenames)

	image = read_cifar10(filename_queue)
	new_image = tf.cast(image.mat, tf.float32)

	height = IMAGE_SIZE
	width = IMAGE_SIZE

	new_image = tf.image.resize_image_with_crop_or_pad(new_image, height, width)
	final_iamge = tf.image.per_image_whitening(new_image)

	min_samples_ratio_in_queue = 0.4
	min_samples_in_queue = int(NUM_EXAMPLES_PER_EPOCH_FOR_EVAL*min_samples_ratio_in_queue)
	return _generate_image_and_label_batch([final_image, image.label],
						min_samples_in_queue, batch_size,
						shuffle_flag=False)

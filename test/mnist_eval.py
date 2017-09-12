#!/usr/bin/python
#-*- coding:utf-8 -*-  
############################  
#File Name: mnist_eval.py
#Author: yang
#Mail: milkyang2008@126.com  
#Created Time: 2017-08-26 17:02:46
############################

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import time
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

import mnist_inference
import mnist_train

#load nearly(last) model every 10s; and test the model accuracy
eval_interval_secs = 10

def evaluate(mnist):
	with tf.Graph().as_default() as g:
		x = tf.placeholder(tf.float32, [None, mnist_inference.input_node],
			name = 'x-input')
		y_ = tf.placeholder(tf.float32, [None, mnist_inference.output_node],
		    name = 'y-input')
		validate_feed = {x: mnist.validation.images,
		                 y_:mnist.validation.labels}

		y = mnist_inference.inference(x,None)

		correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

		variable_averages = tf.train.ExponentialMovingAverage(
			mnist_train.moving_average_decay)
		variables_to_restore = variable_averages.variables_to_restore()
		saver = tf.train.Saver(variables_to_restore)

	#test the model accuracy every 10s
		while True:
			with tf.Session() as sess:
				ckpt = tf.train.get_checkpoint_state(
					mnist_train.model_save_path)
				if ckpt and ckpt.model_checkpoint_path:
					saver.restore(sess, ckpt.model_checkpoint_path)
					global_step = ckpt.model_checkpoint_path\
									  .split('/')[-1].split('-')[-1]
					accuracy_score = sess.run(accuracy,
				    	                      feed_dict=validate_feed)
					print("After %s training step(s), validation "
							"accuracy = %g" % (global_step, accuracy_score))
				else:
					print('No checkpoint file found!')
					return
			time.sleep(eval_interval_secs)

def main(argv=None):
	mnist = input_data.read_data_sets("/tmp/data", one_hot=True)
	evaluate(mnist)

if __name__ == '__main__':
	tf.app.run()

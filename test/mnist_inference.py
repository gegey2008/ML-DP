#!/usr/bin/python
#-*- coding:utf-8 -*-  
############################  
#File Name: mnist_inference.py
#Author: yang
#Mail: milkyang2008@126.com  
#Created Time: 2017-08-26 14:28:23
############################

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf

#define structure and parameter of neural network
input_node = 784
output_node = 10
Layer1_node = 500

def get_weight_variable(shape,regularizer):
	weights = tf.get_variable(
		"weights", shape,
		initializer=tf.truncated_normal_initializer(stddev=0.1))

	if regularizer != None:
		tf.add_to_collection('losses', regularizer(weights))
	return weights

#define forward pass
def inference(input_tensor, regularizer):
	#define layer1
	with tf.variable_scope('layer1'):
		weights = get_weight_variable(
			[input_node, Layer1_node], regularizer)
		biases = tf.get_variable(
			"biases", [Layer1_node], 
			initializer = tf.constant_initializer(0.0))
		layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)
	#define layer2
	with tf.variable_scope('layer2'):
		weights = get_weight_variable(
			[Layer1_node, output_node], regularizer)
		biases = tf.get_variable(
			"biases", [output_node], 
			initializer = tf.constant_initializer(0.0))
		layer2 = tf.matmul(layer1, weights) + biases

	#return layer2
	return layer2


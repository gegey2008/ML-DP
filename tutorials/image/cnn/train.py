#!/usr/bin/python
#-*- coding:utf-8 -*-
############################
#File Name: train.py
#Author: yang
#Mail: milkyang2008@126.com
#Created Time: 2017-09-15 22:05:13
############################

from __future__ import division, print_function, absolute_import

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import cifar10_input
import forward_propagation
import tensorflow as tf
import os
import numpy as np

max_iter_num = 30000
checkpoint_path = './checkpoint'
event_log_path = './event-log'

def train():
	with tf.Graph().as_default():
		global_step = tf.Variable(initial_value=0, trainable=False)
		img_batch, label_batch = cifar10_input.preprocess_input_data()
		logits = forward_propagation.inference(img_batch)
		total_loss = forward_propagation.loss(logits, label_batch)
		one_step_gradient_update = forward_propagation.one_step_train(total_loss, global_step)
		#save
		saver = tf.train.Saver(var_list=tf.global_variables())
		all_summary_obj = tf.summary.merge_all()
		initiate_variables = tf.global_variables_initializer()
		with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
			sess.run(initiate_variables)
			tf.train.start_queue_runners(sess=sess)
			Event_writer = tf.summary.FileWriter(logdir=event_log_path, graph=sess.graph)
			for step in range(max_iter_num):
				_, loss_value = sess.run(fetches=[one_step_gradient_update, total_loss])
				assert not np.isnan(loss_value)
				if step%10 == 0:
				    print('step %d, the loss_value is %.2f' % (step, loss_value))
                                    #f = open("./result.txt",'a') #add  modle
				    #print('step %d, the loss_value is %.2f' % (step, loss_value),file=f)
                                    #f.close()
				if step%100 == 0:
				    all_summaries = sess.run(all_summary_obj)
				    Event_writer.add_summary(summary=all_summaries, global_step=step)
				if step%1000 == 0 or (step+1)==max_iter_num:
				    variables_save_path = os.path.join(checkpoint_path, 'model-parameters.bin')
				    saver.save(sess, variables_save_path, global_step=step)
if __name__ == '__main__':
	train()

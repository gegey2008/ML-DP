#!/usr/bin/python
#-*- coding:utf-8 -*
###########################
#File Name: nin_train.py
#Author: gegey2008
#Mail: milkyang2008@126.com
#Created Time: 2017-11-19 19:15:11
############################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import network_in_network
import numpy as np

max_steps = 30000
checkpoint_path = './checkpoint'
event_log_path = './event-log'


def train():
    with tf.Graph().as_default():
        global_step = tf.Variable(initial_value=0, trainable=False)
        images, labels = network_in_network.distorted_inputs()
        logits = network_in_network.inference(images)
        loss = network_in_network.loss(logits, labels)
        train_op = network_in_network.train(loss, global_step)
        #save
        saver = tf.train.Saver(var_list=tf.global_variables())
        all_summary_obj = tf.summary.merge_all()
        initiate_variables = tf.global_variables_initializer()
        with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
            sess.run(initiate_variables)
            tf.train.start_queue_runners(sess=sess)
            Event_writer = tf.summary.FileWriter(logdir=event_log_path, graph=sess.graph)
            for step in range(max_steps):
                _, loss_value = sess.run(fetches=[train_op, loss])
                assert not np.isnan(loss_value)
                if step%10 == 0:
                    print('step %d, the loss_value is %.2f' % (step, loss_value))
                    f = open("./result.txt",'a') #add  modle
                    print('step %d, the loss_value is %.2f' % (step, loss_value),file=f)
                    f.close()
                if step%100 == 0:
                    all_summaries = sess.run(all_summary_obj)
                    Event_writer.add_summary(summary=all_summaries, global_step=step)
                if step%1000 == 0 or (step+1)==max_steps:
                    variables_save_path = os.path.join(checkpoint_path, 'model-parameters.bin')
                    saver.save(sess, variables_save_path, global_step=step)


if __name__ == '__main__':
    train()



import new
from PIL import Image

import numpy as np
import os
import tensorflow as tf

def conv_layer(input, kernel_size, strides, padding='SAME', force_maxpool=True, name=None):
	with tf.variable_scope(name):
		kernel = tf.get_variable(name='kernel'+name[-1], shape=kernel_size, dtype=tf.float32, initializer=tf.zeros_initializer())
		b = tf.get_variable(name='b'+name[-1], dtype=tf.float32, initializer=tf.constant(999., shape=[kernel_size[-1]]))
		conv = tf.nn.relu((tf.nn.conv2d(input, kernel, strides=strides, padding=padding) + b), name='conv'+name[-1])
		if force_maxpool:	
			return tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding=padding, name='maxpool'+name[-1])
		return conv
	
def flatten_layer(map, name=None):
	shape = map.get_shape()
	with tf.variable_scope(name):
		return tf.reshape(map, [-1, shape[-1]*shape[-2]*shape[-3]])
		
def fc_layer(flatten, output_size, use_relu=True, name=None):
	with tf.variable_scope(name):
		output = tf.get_variable(name='unit'+name[-1], shape=output_size, dtype=tf.float32, initializer=tf.zeros_initializer())
		b = tf.get_variable(name='fc_b'+name[-1], dtype=tf.float32, initializer=tf.constant(999., shape=[output_size[-1]]))
		if use_relu:
			return tf.nn.relu(tf.add(tf.matmul(flatten, output), b), name='relu'+name[-1])
		return tf.nn.softmax(tf.add(tf.matmul(flatten, output), b), name='softmax'+name[-1])

graph = tf.Graph()
with graph.as_default():
	sess = tf.Session(graph=graph)
	
	x = tf.placeholder(tf.float32, shape=[None, 96, 96, 3], name='input_data')
	y = tf.placeholder(tf.float32, shape=[None], name='real_label')
	y_onehot = tf.placeholder(tf.float32, shape=[None, 10], name='onehot_real_label')
	
	with tf.variable_scope('FEATURE_EXTRACTOR'):
		conv_l1 = conv_layer(x, [5, 5, 3, 24], [1, 1, 1, 1], name='CONV_L1')
		conv_l2 = conv_layer(conv_l1, [3, 3, 24, 48], [1, 1, 1, 1], name='CONV_L2')
		#conv_l3 = conv_layer(conv_l2, [3, 3, 32, 32], [1, 1, 1, 1], name='CONV_L3')
		conv_l3 = conv_layer(conv_l2, [2, 2, 48, 72], [1, 1, 1, 1], name='CONV_L3')
		conv_l4 = conv_layer(conv_l3, [2, 2, 72, 180], [1, 1, 1, 1], name='CONV_L4')
		flatten = flatten_layer(conv_l4, 'FLATTEN_L')
		fc_l1 = fc_layer(flatten, [6480, 1536], name='FC_L1')
		keep_prob = tf.placeholder(tf.float32)
		drop_l = tf.nn.dropout(fc_l1, keep_prob)
		fc_l2 = fc_layer(drop_l, [1536, 512], name='FC_L2')
		l2_norm_l = tf.nn.l2_normalize(fc_l2, axis=1)

	with tf.variable_scope('CLASSIFIER'):
		fc_l3 = fc_layer(fc_l2, [512, 256], name='FC_L3')
		softmax = fc_layer(fc_l3, [256, 10], False, 'FC_SOFTMAX_L4')
	#sess.run(tf.global_variables_initializer())
	'''checkpoint = tf.train.get_checkpoint_state("C:\\work\\face\\model")
	meta_graph_path = checkpoint.model_checkpoint_path + ".meta"'''
	
	
	'''restore = tf.train.import_meta_graph("C:\\tmp\\model\\model.ckpt-0.meta")
	restore.restore(sess, tf.train.latest_checkpoint("C:\\tmp\\model"))'''
	
	saver = tf.train.Saver()
	saver.restore(sess, tf.train.latest_checkpoint("C:\\tmp\\model"))
	
	all_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'CLASSIFIER')
	for v in all_vars:
		v_ = sess.run(v)
		print(v)
		print(v_)
		print()


	'''x = graph.get_tensor_by_name("input_data:0")
	keep_prob = graph.get_tensor_by_name("FEATURE_EXTRACTOR/Placeholder:0")
	test = graph.get_tensor_by_name("CLASSIFIER/FC_SOFTMAX_L4/softmax4:0")'''
	'''files, labels = read_dataset('C:\\Users\\nngbao\\Downloads\\face_triplet\\data\\face')
	a = DataSet(files, labels)
	#print(batch_x)
	#print(test)
	print("KLDJGIORDJHGIOREJHGIOREHGIOrjGIOWEJGIOWEJGIOJEIOFJWIOEJFWIOEF")
	for i in range(50):
		batch_x, batch_y = a.next_batch(1)
		print(batch_x[-1])
		res = sess.run([softmax], feed_dict={x: [np.asarray(Image.open(batch_x[-1])) / 255.], keep_prob: 1.})
		print(res)
		print('true: ', batch_y[-1])
		print()'''
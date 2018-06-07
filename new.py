from PIL import Image
import numpy as np
import os
import tensorflow as tf

def read_dataset(root):
	i = -1
	files = []
	labels = []
	for dir in os.listdir(root):
		dir = os.path.join(root, dir)
		if os.path.isdir(dir):
			i += 1
			for file in os.listdir(dir):
				file = os.path.join(dir, file)
				if file[-3:] == 'png' or file[-3:] == 'jpg':
					files.append(file)
					labels.append(i)
	return files, labels

class DataSet(object):
	def __init__(self, files, labels):
		self.flag = 0
		combined = list(zip(files, labels))
		np.random.shuffle(combined)
		self.files, self.labels = zip(*combined)

	def get_files(self):
		return self.files
		
	def get_labels(self):
		return self.labels
		
	def next_batch(self, batch_size):
		start = self.flag
		self.flag += batch_size
		if self.flag > len(self.labels):
			start = 0
			self.flag = batch_size
			combined = list(zip(self.files, self.labels))
			np.random.shuffle(combined)
			self.files, self.labels = zip(*combined)
		end = self.flag
		return self.files[start:end], self.labels[start:end]
		

def conv_layer(input, kernel_size, strides, padding='SAME', force_maxpool=True, name=None):
	with tf.variable_scope(name):
		kernel = tf.get_variable(name='kernel'+name[-1], shape=kernel_size, dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.1))
		b = tf.get_variable(name='b'+name[-1], dtype=tf.float32, initializer=tf.constant(0.1, shape=[kernel_size[-1]]))
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
		output = tf.get_variable(name='unit'+name[-1], shape=output_size, dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.1))
		b = tf.get_variable(name='fc_b'+name[-1], dtype=tf.float32, initializer=tf.constant(0.1, shape=[output_size[-1]]))
		if use_relu:
			return tf.nn.relu(tf.add(tf.matmul(flatten, output), b), name='relu'+name[-1])
		return tf.nn.softmax(tf.add(tf.matmul(flatten, output), b), name='softmax'+name[-1])
		
files, labels = read_dataset('C:\\Users\\nngbao\\Downloads\\face_triplet\\data\\face')
a = DataSet(files, labels)
t1 = a.get_files()
t2 = a.get_labels()
'''print(t1[5])
print(t2[5])'''

image = np.asarray(Image.open(t1[5])) / 255.
g = tf.Graph()
with g.as_default():
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
		print(softmax)
	
	triplet_loss = tf.contrib.losses.metric_learning.triplet_semihard_loss(y, l2_norm_l)
	cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_onehot, logits=softmax))
	total_loss = triplet_loss + cross_entropy_loss
	accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(softmax, 1), tf.argmax(y_onehot, 1)), tf.float32))
	first_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "FEATURE_EXTRACTOR")
	second_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "CLASSIFIER")
	
	global_step = tf.Variable(0, trainable=False)
	learning_rate = tf.where(tf.greater_equal(global_step, 8000),
							tf.train.polynomial_decay(1e-3, global_step,
														8000, 4e-6,
														power=1.0),
							1e-4)
	tf.summary.scalar('learning_rate', learning_rate)
	
	first_train_ops = tf.train.AdamOptimizer(learning_rate).minimize(triplet_loss, global_step=global_step, var_list=first_train_vars)
	second_train_ops = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_loss, var_list=second_train_vars)
	
	tf.summary.scalar("triplet_loss", triplet_loss)
	tf.summary.scalar("cross_entropy_loss", cross_entropy_loss)
	tf.summary.scalar("total_loss", total_loss)
	
	merged_summary_op = tf.summary.merge_all()
	
	summary_writer = tf.summary.FileWriter('/tmp', graph=tf.get_default_graph())
	saver = tf.train.Saver()
	
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	
	
	for step in range (28000):
		batch_x, batch_y = a.next_batch(40)
		_x = []
		_y_onehot = []
		for j in range (40):
			onehot = np.zeros(10)
			onehot[batch_y[j]] = 1
			_y_onehot.append(onehot)
			image_array = np.asarray(Image.open(batch_x[j])) / 255.
			_x.append(image_array)
		batch_x = np.asarray(_x)
		batch_y = np.asarray(batch_y)
		_y_onehot = np.asarray(_y_onehot)

		_, _, _triplet_loss, _cross_entropy_loss, summary = sess.run([first_train_ops, second_train_ops, triplet_loss, cross_entropy_loss, merged_summary_op], feed_dict={x: batch_x, y: batch_y, y_onehot: _y_onehot, keep_prob: 0.55})
		if step % 100 == 0:
			train_accuracy = sess.run([accuracy], feed_dict={x: batch_x, y_onehot: _y_onehot, keep_prob: 1.})
			#train_accuracy = accuracy.eval({x: batch_x, y: batch_y, keep_prob: 1.0}, sess)
			print('global_step %s, triplet_loss %.4f, cross_entropy_loss %.4f, total_loss %.4f, accuracy %.2f' % (tf.train.global_step(sess, global_step), _triplet_loss, _cross_entropy_loss, (_triplet_loss + _cross_entropy_loss), train_accuracy[-1]))
#			print(', triplet_loss %f' % (step, _triplet_loss))
			summary_writer.add_summary(summary, step)
			summary_writer.flush()
		
	save_path = saver.save(sess, '/tmp/model' + "/model.ckpt", global_step=step)
	#logging.info("Model saved in file: %s" % save_path)
	
	'''print(conv_l1)
	print(conv_l2)
	print(conv_l3)
	print(conv_l4)
	print(conv_l5)
	print(flatten)
	print(fc_l1)
	print(fc_l2)
	print(triplet_loss)'''

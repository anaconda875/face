import new
from PIL import Image

import numpy as np
import os
import tensorflow as tf


graph = tf.Graph()
with graph.as_default():
	sess = tf.Session(graph=graph)
	'''checkpoint = tf.train.get_checkpoint_state("C:\\work\\face\\model")
	meta_graph_path = checkpoint.model_checkpoint_path + ".meta"'''
	restore = tf.train.import_meta_graph("C:\\tmp\\model\\model.ckpt-27999.meta")
	restore.restore(sess, tf.train.latest_checkpoint("C:\\tmp\\model"))
	x = graph.get_tensor_by_name("input_data:0")
	keep_prob = graph.get_tensor_by_name("FEATURE_EXTRACTOR/Placeholder:0")
	test = graph.get_tensor_by_name("CLASSIFIER/FC_SOFTMAX_L4/softmax4:0")
	files, labels = new.read_dataset('C:\\Users\\nngbao\\Downloads\\face_triplet\\data\\face')
	a = new.DataSet(files, labels)
	#print(batch_x)
	#print(test)
	for i in range(50):
		batch_x, batch_y = a.next_batch(1)
		print(batch_x[-1])
		res = sess.run([test], feed_dict={x: [np.asarray(Image.open(batch_x[-1])) / 255.], keep_prob: 1.})
		print(res)
		print('true: ', batch_y[-1])
		print()
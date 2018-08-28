from PIL import Image
import tensorflow as tf
import numpy as np
import os

with tf.gfile.GFile('C:\\Users\\nngbao\\out.pb', "rb") as f:
	graph_def = tf.GraphDef()
	graph_def.ParseFromString(f.read())
	
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
	
with tf.Graph().as_default() as graph:
	tf.import_graph_def(graph_def, name="")
	x = graph.get_tensor_by_name("input_data:0")
	keep_prob = graph.get_tensor_by_name("FEATURE_EXTRACTOR/Placeholder:0")
	test = graph.get_tensor_by_name("FEATURE_EXTRACTOR/l2_normalize:0")
	#files, labels = read_dataset('./face')
	files, labels = read_dataset('./face')
	a = DataSet(files, labels)
	sess = tf.Session(graph=graph)
	for i in range(1):
		#batch_x, batch_y = a.next_batch(1)
		a = np.array(sess.run([test], feed_dict={x: [np.asarray(Image.open('./face/train/lam truong/lam truong73.png')) / 255.], keep_prob: 1.}))
		#print(batch_x[-1])
		#print(res1)
		b = np.array(sess.run([test], feed_dict={x: [np.asarray(Image.open('./face/train/lam truong/lam truong66.png')) / 255.], keep_prob: 1.}))
		dist = np.linalg.norm(a-b)
		print(dist)
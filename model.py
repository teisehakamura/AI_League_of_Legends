import tensorflow as tf
import numpy as np
import os

from tf_load_image import Load_Image

flags = tf.app.flags

flags.DEFINE_string("filename", "LOL_data.npy", "Please write down the name of npy data")
flags.DEFINE_integer("width", 224, "Width")
flags.DEFINE_integer("height", 224, "Height")
flags.DEFINE_integer("number_of_class", 3, "How many classes do you have?")
flags.DEFINE_integer("training_epoch", 1000, "epoch")
flags.DEFINE_integer("batch_size", 32, "batch_size")
flags.DEFINE_string("TB_CP", "./log", "tensorboard and checkpoint")


FLAGS = flags.FLAGS

path = os.path.join(os.getcwd() + "/data/")

class Model:
	def __init__(self, sess, name):
		self.sess = sess
		self.name = name
		self.summary = self.__build__()

	def __build__(self):
		with tf.variable_scope(self.name):
			self.training = tf.placeholder(tf.bool)
			self.x = tf.placeholder(tf.float32, [None, FLAGS.width, FLAGS.height, 3])
			x_image = tf.reshape(self.x, [-1,FLAGS.width,FLAGS.height,3])

			self.y = tf.placeholder(tf.int32, [None, FLAGS.number_of_class])

			#cnn
			# 224,224,3 -> 224,224,32 -> 112,112,32
			conv2d1 = tf.layers.conv2d(inputs=x_image, filters= 32, kernel_size = [3,3],
				padding = 'same', activation = 'relu')
			max_pooling2d1 = tf.layers.max_pooling2d(inputs = conv2d1, pool_size = [3,3],
				strides = 2,padding = 'same')
			dropout1 = tf.layers.dropout(inputs=max_pooling2d1,
				rate=0.5, training= self.training)

			#112,112,32, -> 112,112,64, -> 56,56,64
			conv2d2 = tf.layers.conv2d(inputs=dropout1,
				filters=64, kernel_size=[3,3], strides= 1, padding='same', activation ='relu')
			max_pooling2d2 = tf.layers.max_pooling2d(inputs=conv2d2, 
				pool_size=[3,3],strides =2, padding = 'same')
			dropout2 = tf.layers.dropout(inputs=max_pooling2d2,
				rate=0.5, training=self.training)

			#56 56,64 -> 56,56,128 -> 28,28,128
			conv2d3 = tf.layers.conv2d(inputs= dropout2,
				filters =128, kernel_size=[3,3], strides=1, padding='same', activation = 'relu')
			max_pooling2d3 = tf.layers.max_pooling2d(inputs=conv2d3,
				pool_size=[3,3], strides=2, padding='same')
			dropout3 = tf.layers.dropout(inputs= max_pooling2d3,
				rate=0.5, training =self.training)
			#28,28 128 -> 28 28 256 -> 14 14 256
			conv2d4 = tf.layers.conv2d(inputs= dropout3, filters = 256,
				kernel_size = [3,3], strides =1, padding= 'same', activation= 'relu')
			max_pooling2d4 = tf.layers.max_pooling2d(inputs = conv2d4,
				pool_size= [3,3], strides =2, padding = 'same')
			dropout4 = tf.layers.dropout(inputs =max_pooling2d4,
				rate =0.5, training = self.training)
			# 14 14 256 14 14 512 7 7 512
			conv2d5 = tf.layers.conv2d(inputs= dropout4, filters = 512,
				kernel_size = [3,3], strides =1, padding= 'same', activation= 'relu')
			max_pooling2d5 = tf.layers.max_pooling2d(inputs = conv2d5,
				pool_size= [3,3], strides =2, padding = 'same')
			dropout5 = tf.layers.dropout(inputs =max_pooling2d5,
				rate =0.5, training = self.training)

			#flatten dense625 125 3
			flatten1 = tf.reshape(dropout5, [-1, 7*7*512])
			dense1 = tf.layers.dense(inputs=flatten1,
				units =625, activation= 'relu')
			dense2 = tf.layers.dense(inputs=dense1,
				units = 125, activation = 'relu')
			self.logits = tf.layers.dense(inputs=dense2,
				units=FLAGS.number_of_class, activation = 'relu')

		# cost function = sigma log(logits) - y(real number) * 1/n 
		self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,
			labels = self.y))
		self.optimizer = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(self.cost)
		tf.summary.scalar("cost", self.cost)

		prediction =tf.equal(
			tf.argmax(self.logits, 1), tf.argmax(self.y, 1))
		self.accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
		tf.summary.scalar("accuracy", self.accuracy)
		self.summary = tf.summary.merge_all()
		return self.summary

	def train(self, x_train, y_train, training = True):
		return self.sess.run([self.cost, self.optimizer, self.summary],
			feed_dict={
			self.x: x_train, self.y:y_train, self.training: training
			})

	def get_accuracy(self, x_eval , y_eval, training=False):
		return self.sess.run([self.accuracy],
			feed_dict = {
			self.x:x_eval, self.y:y_eval, self.training: training
			})

	def test(self, x_test, trainint = False):
		return self.sess.run([self.logits],
			feed_dict = {
			self.x: x_test, self.training: training
			})

def batch_function(x_train, y_train, batch_size):
	for start_idx in range(0, len(x_train) - batch_size +1, batch_size):
		excerpt = slice(start_idx, start_idx + batch_size)
		yield x_train[excerpt], y_train[excerpt]

def main(_):

	#sess
	sess = tf.Session()
	model = Model(sess, "model1")

	sess.run(tf.global_variables_initializer())
	print("Convolution Start")

	#data
	x_train, y_train, x_eval, y_eval = Load_Image(path + FLAGS.filename).__build__()
	print(y_train.shape)
	print(x_train.shape)
	#TB
	writer = tf.summary.FileWriter(FLAGS.TB_CP)
	writer.add_graph(sess.graph)
	global_step = 0

	#CP
	saver = tf.train.Saver()
	checkpoint = tf.train.get_checkpoint_state(FLAGS.TB_CP)
	#load model
	# if checkpoint and checkpoint.model_checkpoint_path:
	# 	try:
	# 		saver.restore(sess, checkpoint.model_checkpoint_path)
	# 		print("Sucessfully loaded:", checkpoint.model_checkpoint_path)
	# 		print("accuracy", model.get_accuracy(x_eval, y_eval))
	# 	except:
	# 		print("Error")
	# else:
	# 	print("Coundn't find it")

	for epoch in range(FLAGS.training_epoch):
		train_cost, train_opt, n_batch = 0,0,0

		for x_train_batch, y_train_batch in batch_function(x_train, y_train, FLAGS.batch_size):
			cost, acc, summary = model.train(x_train_batch, y_train_batch)
			train_cost += cost; n_batch +=1
			#TB
			writer.add_summary(summary, global_step=global_step)
			global_step +=1
			#CP
		saver.save(sess, FLAGS.TB_CP + "/model", global_step= global_step)
		print("avg cost:", np.sum(train_cost)/ n_batch)
		print("accuracy:", model.get_accuracy(x_eval, y_eval))
	print("End")
	
if __name__ == "__main__":
	tf.app.run()
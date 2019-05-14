import numpy as np
import tensorflow as tf
import argparse
import os

from sklearn.preprocessing import OneHotEncoder
from tf_load_image import Load_Image


parser = argparse.ArgumentParser(description = "pointer")

parser.add_argument("--filename",
    type = str,
    default = "data/LOL_data.npy",
    help = "the file to save images and keyboard and coordinates")
parser.add_argument("--width",
    type = int,
    default = 224,
    help = "Width")
parser.add_argument("--height",
    type = int,
    default = 224,
    help = "Width")
parser.add_argument("--BATCH_SIZE",
    type = int,
    default = 32,
    help = "Width")
parser.add_argument("--EPOCHS",
    type = int,
    default = 10,
    help = "Width")

args = parser.parse_args()

filename = args.filename
width = args.width
height = args.height
EPOCHS = args.EPOCHS
BATCH_SIZE = args.BATCH_SIZE

batch_size = tf.placeholder(tf.int64)

x = tf.placeholder(tf.float32, [None, args.width, args.height, 3])
y = tf.placeholder(tf.int32, [None, 3])

train_dataset = tf.data.Dataset.from_tensor_slices((x,y)).batch(batch_size).repeat()
test_dataset = tf.data.Dataset.from_tensor_slices((x,y)).batch(batch_size) # always batch even if you want to one shot it

image_data, label_data, train_data, test_data = Load_Image.dataset(args.filename)

iter = tf.data.Iterator.from_structure(train_dataset.output_types,
                                           train_dataset.output_shapes)
features, labels = iter.get_next()

train_init_op = iter.make_initializer(train_dataset)
test_init_op = iter.make_initializer(test_dataset)

flatten = tf.reshape(features, [-1, 224*224*3])
net = tf.layers.dense(flatten, 8, activation=tf.tanh) # pass the first value from iter.get_next() as input
net = tf.layers.dense(net, 8, activation=tf.tanh)
prediction = tf.layers.dense(net, 3, activation=tf.tanh)
print("prediction", prediction)

loss = tf.losses.mean_squared_error(prediction, labels) # pass the second value from iter.get_net() as label
train_op = tf.train.AdamOptimizer().minimize(loss)

n_batches = args.width*args.height*3 // 32

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # initialise iterator with train data
    sess.run(train_init_op, feed_dict = {x : image_data, y: label_data, batch_size: 16})
    print('Training...')
    for i in range(args.EPOCHS):
        tot_loss = 0
        for _ in range(n_batches):
            _, loss_value = sess.run([train_op, loss])
            tot_loss += loss_value
        print("Iter: {}, Loss: {:.4f}".format(i, tot_loss / n_batches))
    # initialise iterator with test data
    sess.run(test_init_op, feed_dict = {x : image_data, y: label_data, batch_size:len(test_data[0])})
    print('Test Loss: {:4f}'.format(sess.run(loss)))
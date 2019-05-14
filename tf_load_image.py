import numpy as np
import tensorflow as tf
import argparse
from sklearn.preprocessing import OneHotEncoder

import os


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

args = parser.parse_args()

filename = args.filename
width = args.width
height = args.height

class Load_Image:
	def __init__(self, filename):
		self.filename = filename
		self.__build__()

	def __build__(self):
		assert filename, "data file is missing"
		
		data = np.load(filename)

		image_data, label_data = Load_Image.preprocessing(data)

		x_train, y_train, x_eval, y_eval = Load_Image.seperate(image_data, label_data)

		return x_train, y_train, x_eval, y_eval


	def preprocessing(data):
		image_data = []
		label_data = []

		for x_data, y_data in data:
			image_data.append(x_data)
			label_data.append(y_data)
		image_data, label_data = np.array(image_data), np.array(label_data).reshape(-1, 1)
		image_data = np.array(image_data).reshape(len(image_data[-1,:]))
		image_data = np.array([i for i in image_data])

		#OneHotEncoder	
		encoder = OneHotEncoder()
		encoder.fit(label_data)

		label_data = encoder.transform(label_data).toarray()

		return image_data, label_data

	def seperate(image_data, label_data):

		train_size = round(int(len(label_data) * 0.8))

		x_train = image_data[0:train_size]
		x_eval = image_data[train_size:]
		y_train = label_data[0:train_size]
		y_eval = label_data[train_size:]

		return x_train, y_train, x_eval, y_eval


if __name__ == "__main__":
	x_train, y_train, x_eval, y_eval = Load_Image(filename = filename).__build__()

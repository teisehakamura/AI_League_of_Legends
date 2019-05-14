import tensorflow as tf
import argparse
import cv2

import os
import numpy as np
from PIL import ImageGrab

#argparse
parser = argparse.ArgumentParser(description = "Save Image")

parser.add_argument("--filename",
	type = str,
	default = "LOL_Data.npy",
	help = "Save 500 images on Tower, Minion, and Ezreal")
parser.add_argument("--width",
	type = int,
	default = 224,
	help = "Width")
parser.add_argument("--height",
	type = int,
	default = 224,
	help = "Width")

parser.add_argument("--part_1",
    type = int,
    default = 500,
    help = "500 Images of Tower")
parser.add_argument("--part_2",
    type = int,
    default = 1000,
    help = "500 Images of Minion")
parser.add_argument("--part_3",
    type = int,
    default = 1500,
    help = "500 Images of Ezreal")

args = parser.parse_args()

filename = args.filename
width = args.width
height = args.height
part_1 = args.part_1
part_2 = args.part_2
part_3 = args.part_3
# tf.app.flags

# Additionally also can use tf.app.flags instead of argparse

flags = tf.app.flags

flags.DEFINE_string("filename", "LOL_Data", "Save 500 images on Tower, Minion, and Ezreal")
flags.DEFINE_integer("width", 224, "Width")
flags.DEFINE_integer("height", 224, "height")

FLAGS = flags.FLAGS

data = os.path.join(os.getcwd(), "data")
images = os.path.join(os.getcwd(), "images")
if not os.path.exists(data):
	os.mkdir(data)

if not os.path.exists(images):
	os.mkdir(images)

class Save_Image:
	def __init__(self, width, height, filename, part_1, part_2,part_3):
		self.width = width
		self.height = height
		self.filename = filename
		self.part_1 = part_1
		self.part_2 = part_2
		self.part_3 = part_3
		self.__build__()
	def __build__(self):
		count = 0
		data_set = []
		label_set = []
		train_data = []

		assert filename, "filename is missing"

		while(True):
			count += 1
			warm_up = 30

			if count % 10 == 0:
				print(count)
			time = cv2.getTickCount()
			printscreen = np.array(ImageGrab.grab(bbox = (0, 400, 1000, 1200)))
			

			printscreen = cv2.cvtColor(printscreen, cv2.COLOR_BGR2RGB)
			printscreen = cv2.resize(printscreen, (self.width,self.height))
			# print(printscreen.shape)
			cv2.imshow("window", printscreen) 
		
			# Save images
			Save_Image.create_save_image(count, printscreen, warm_up)
			
			# Save .npy or .pickle (-1, 224, 224, 3)
			printscreen = np.array(printscreen).reshape(self.width, self.height, 3)

			if count <= part_1 + warm_up and count > warm_up: # 30 530
				tower = 1
				data_set.append(printscreen)
				label_set.append(tower)
			elif count > part_1 + (2*warm_up) and count <= part_2 + (2*warm_up): # 560 1060
				minion = 2
				data_set.append(printscreen)
				label_set.append(minion)
			elif count > part_2 + (3*warm_up) and count <= part_3 + (3*warm_up): # 1090 1590
				ezreal = 3
				data_set.append(printscreen)
				label_set.append(ezreal)
			elif count >= part_3 + (3*warm_up): #1590
				train_data.append([data_set, label_set])
				train_data = np.array(train_data)
				np.save("data/" + filename, train_data)
				print("LOL_Data.npy saved", "shape is {}".format(train_data.shape))
				break

			if cv2.waitKey(25) & 0xFF == ord("q"):
				cv2.destroyAllWindow()
				break

	def create_save_image(count, printscreen, warm_up):

		if count <= warm_up: # 30
			# print("tower")
			print("warm up time :" + str(warm_up-count))
		elif count <= part_1 + warm_up and count > warm_up: # 30 530
			print("A picture is taken")
			champ1 = "tower"
			cv2.imwrite("images/" + champ1 + "/" + champ1 + "." + str(count) + ".jpg", printscreen)
		elif count > part_1 + warm_up and count <= part_1 + (2*warm_up): # 530 560
			# print("minion")
			print("warm up time :" + str(part_1+(2*warm_up)-count))
		elif count > part_1 + (2*warm_up) and count <= part_2 + (2*warm_up): # 560 1060
			print("A picture is taken")
			champ2 = "minion"
			cv2.imwrite("images/" + champ2 + "/" + champ2 + "." + str(count) + ".jpg", printscreen)
		elif count > part_2 + (2*warm_up) and count <= part_2 + (3*warm_up): #1060 1090
			# print("ezreal")
			print("warm up time: " + str(part_2+(3*warm_up)-count))
		elif count > part_2 + (3*warm_up) and count <= part_3 + (3*warm_up): # 1090 1590
			print("A picture is taken")
			champ3 = "ezreal"
			cv2.imwrite("images/" + champ3 + "/" + champ3 + "." + str(count) + ".jpg", printscreen)
		
if __name__ == "__main__":
	#tf.app.run()
	Save_Image(width = width, height = height, filename = filename, part_1 = part_1, part_2 = part_2, part_3 = part_3)
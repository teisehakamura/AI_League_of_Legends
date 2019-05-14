# Incubit_Project-AI_League_Of_Legends-

# Explaination (Classification and Object Detection)

1. I respectively save 500 images of Tower, Minion, and Ezreal on League of Legends (Total 3 classes)
- Used parser or tf.run.flags to insert width, height, and filename
- Used cv2 to take each 500 pictures
- Saved those pictures with np.save(The size of .npy is 233.1 mb)

!Screen Shot 2019-05-15 at 1.30.52(./git/Screen Shot 2019-05-15 at 1.30.52.png)

2. Model - Used AWS to use GPU that has p2.x2large to operate model. I have used 3500 epochs


3. Tensorboard -

4. Result -

# Installation

1. You can install some packages, tensorflow, argparse, pillow, and numpy

pip install -r requirement.txt

# How to Use Classification and Object Detection

2. To create .npy and 500 pictures, use tf_save_image.py

python tf_save_image.py --filename LOL_data.npy

3. If you run it, the file LOL_data.npy on the folder of data is created and 1,500 images are saved on the image folder of images.

4. To seperate LOL_data.npy into two groups, train_data and image_data, you might as well use tf_load_image.py

python tf_load_image.py --filename LOL_data.npy

4. To create a model, you might use model.py that has CNN mode

python model.py


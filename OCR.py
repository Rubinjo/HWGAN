import numpy as np
import os
from pathlib import Path
import sys

from PIL import Image
from cv2 import cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from imutils import build_montages

import tensorflow as tf
from tensorflow.keras import optimizers
# import tensorflow_datasets as tfds
from tensorflow.keras import models, layers
from tensorflow.keras import utils

# Variables
# Model parameters
EPOCHS = 20
BATCH_SIZE = 200
LEARNING_RATE = 0.01
# Dataset distribution
TEST_SIZE = 0.3
# Image dimensions
WIDTH = 28
HEIGHT = 28

# Set tensorflow warning level
tf.get_logger().setLevel('INFO')

# Create empty lists to hold the data
images = []
labels = []
labelsIdx = []

# Dataset folder
rootdir = Path("./dataset/EMNIST/")

# List current package versions
print("You are using Python version: " + sys.version)
print("You are using Tensorflow version: " + tf.__version__)

# List if there is/are available GPU(s)
gpu = tf.config.list_physical_devices('GPU')
num_gpu = len(gpu)
if num_gpu > 1:
	print("\n" + num_gpu + "GPUs were found")
elif num_gpu > 0:
	print("\n1 GPU was found")
else:
	print("\nNo GPU was found")

print("\nLoading in data...")

# Loop through all directories and files
i = 1
for subdir in rootdir.iterdir():
	while i < 2:
		for root, dirs, files in os.walk(subdir):
			for name in files:
				path = os.path.join(root, name)
				try:
					# Check if file is valid image
					v_image = Image.open(path)
					v_image.verify()
					# Read original image
					image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
					# Resize image
					image = cv2.resize(image, (WIDTH, HEIGHT))
					# Add image to data list
					images.append(image)
					# Get label out of folder name
					labels_hex = os.path.basename(os.path.normpath(subdir))
					labels_bytes = bytes.fromhex(labels_hex)
					labels_ascii = labels_bytes.decode("ASCII")
					# Add label to data list
					labels.append(labels_ascii)
				except Exception:
					print("file " + path + " is corrupt or not a .png file and has been skipped.")
					continue
		i += 1
	break

print("Converting data...")

# Convert labels to numerical index labels
for label in labels:
	idx = ord(label)
	labelsIdx.append(idx)

# Convert list to numpy array for further usage
images = np.array(images)
labelsIdx = np.array(labelsIdx)

# load data
(X_train, X_test, y_train, y_test) = train_test_split(images, labelsIdx, test_size=TEST_SIZE)
# reshape to be [samples][width][height][channels]
X_train = X_train.reshape((X_train.shape[0], WIDTH, HEIGHT, 1)).astype('float32')
X_test = X_test.reshape((X_test.shape[0], WIDTH, HEIGHT, 1)).astype('float32')
# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255
# one hot encode outputs
y_train = utils.to_categorical(y_train)
y_test = utils.to_categorical(y_test)
num_classes = y_test.shape[1]

print("Data has been processed\n")

# define the larger model
def make_recognizer_model():
	# create model
	model = models.Sequential()
	model.add(layers.Conv2D(30, (5, 5), input_shape=(WIDTH, HEIGHT, 1), activation='relu'))
	model.add(layers.MaxPooling2D())
	model.add(layers.Conv2D(15, (3, 3), activation='relu'))
	model.add(layers.MaxPooling2D())
	model.add(layers.Dropout(0.2))
	model.add(layers.Flatten())
	model.add(layers.Dense(128, activation='relu'))
	model.add(layers.Dense(50, activation='relu'))
	model.add(layers.Dense(num_classes, activation='softmax'))
	return model
# build the model
model = make_recognizer_model()
# Create optimizer
opt = optimizers.Adam(learning_rate=LEARNING_RATE)
# Compile model
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
# Fit the model
fit = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=EPOCHS, batch_size=BATCH_SIZE)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Large CNN Error: %.2f%%" % (100-scores[1]*100))
# Save the model
model.save('models/ocr_model')

# construct a plot that plots the training history
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, fit.history["loss"], label="train_loss")
plt.plot(N, fit.history["val_loss"], label="val_loss")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
# save plot image
plt.savefig('models/ocr_model/ocr_graph.png')

# initialize our list of output test images
images = []
# randomly select a few testing characters
for i in np.random.choice(np.arange(0, len(y_test)), size=(49,)):
	# classify the character
	probs = model.predict(X_test[np.newaxis, i])
	prediction = probs.argmax(axis=1)
	label = chr(prediction[0])
	# extract the image from the test data and initialize the text
	# label color as green (correct)
	image = (X_test[i] * 255).astype("uint8")
	color = (0, 255, 0)
	# otherwise, the class label prediction is incorrect
	# label color as red (incorrect)
	if prediction[0] != np.argmax(y_test[i]):
		color = (0, 0, 255)
	# merge the channels into one image, resize the image to 96 x 96
	# draw the predicted label on the image
	image = cv2.merge([image] * 3)
	image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_LINEAR)
	cv2.putText(image, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
		color, 2)
	# add the image to our list of output images
	images.append(image)
# construct the montage for the images
montage = build_montages(images, (96, 96), (7, 7))[0]
# show the output montage
cv2.imwrite("models/ocr_model/ocr_example.png", montage)
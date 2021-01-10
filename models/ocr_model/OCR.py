import os
import numpy as np
from pathlib import Path
import sys

from PIL import Image
from cv2 import cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from imutils import build_montages

import tensorflow as tf
from tensorflow.keras import models, layers, utils, optimizers

class OCR:
	def __init__(self, number_epochs = 50, batch_size = 128, learning_rate = 0.001, width = 28, height = 28):
		self.N_EPOCHS = number_epochs
		self.N_BATCH = batch_size
		self.LR = learning_rate
		self.WIDTH = width
		self.HEIGHT = height
		self.model = models.load_model("./models/ocr_model/ocr_model.h5", compile=False)
	
	def define_recognizer(self, alphabet):
		# Create neural network layers
		model = models.Sequential()
		model.add(layers.Conv2D(30, (5, 5), input_shape=(self.WIDTH, self.HEIGHT, 1), activation='relu'))
		model.add(layers.MaxPooling2D())
		model.add(layers.Conv2D(15, (3, 3), activation='relu'))
		model.add(layers.MaxPooling2D())
		model.add(layers.Dropout(0.2))
		model.add(layers.Flatten())
		model.add(layers.Dense(128, activation='relu'))
		model.add(layers.Dense(50, activation='relu'))
		model.add(layers.Dense(len(alphabet) + 1, activation='softmax'))
		# Create optimizer
		opt = optimizers.Adam(learning_rate=self.LR)
		# Compile model
		model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
		return model

	def predict(self, char):
		return self.model.predict(char[np.newaxis, 0])

	def evaluate(self, r_model, fit, images_test, labels_test, num_classes, alphabet):
		# Evaluate accuracy of the model
		scores = r_model.evaluate(images_test, labels_test, verbose=0)
		print("Accuracy: %.2f%%" % (scores[1]*100))

		# construct a plot that plots the training history
		N = np.arange(0, self.N_EPOCHS)
		plt.style.use("ggplot")
		plt.figure()
		plt.plot(N, fit.history["loss"], label="train_loss")
		plt.plot(N, fit.history["val_loss"], label="val_loss")
		plt.title("Training Loss")
		plt.xlabel("Epoch #")
		plt.ylabel("Loss")
		plt.legend(loc="lower left")
		# save plot image
		plt.savefig("./models/ocr_model/ocr_graph.png")
		plt.close()

		# initialize our list of output test images
		images = []
		# randomly select a few testing characters
		for i in np.random.choice(np.arange(0, len(labels_test)), size=(49,)):
			# classify the character
			probs = r_model.predict(images_test[np.newaxis, i])
			prediction = probs.argmax(axis=1)
			label = alphabet[prediction[0] - 1]
			# extract the image from the test data and initialize the text
			# label color as green (correct)
			image = (images_test[i] * 255).astype("uint8")
			color = (0, 255, 0)
			# otherwise, the class label prediction is incorrect
			# label color as red (incorrect)
			if prediction[0] != np.argmax(labels_test[i]):
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
		cv2.imwrite("./models/ocr_model/ocr_example.png", montage)
	
	def train(self, r_model, alphabet, images_train, labels_train, images_test, labels_test, evaluation = True):
		# reshape to be [samples][width][height][channels]
		images_train = np.expand_dims(images_train, axis=-1).astype('float32')
		images_test = np.expand_dims(images_test, axis=-1).astype('float32')
		# normalize inputs from 0-255 to 0-1
		images_train = images_train / 255
		images_test = images_test / 255
		# one hot encode outputs
		labels_train = utils.to_categorical(labels_train)
		labels_test = utils.to_categorical(labels_test)
		num_classes = labels_test.shape[1]
		# Fit the model
		fit = r_model.fit(images_train, labels_train, validation_data=(images_test, labels_test), epochs=self.N_EPOCHS, batch_size=self.N_BATCH)
		# Save model
		r_model.save("./models/ocr_model/ocr_model.h5")
		if evaluation == True:
			self.evaluate(r_model, fit, images_test, labels_test, num_classes, alphabet)
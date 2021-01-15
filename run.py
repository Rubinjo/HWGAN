# Set tensorflow warning level
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from PIL import Image, ImageOps
from cv2 import cv2
import numpy as np

import sys
from helper.userinput import *

import tensorflow as tf
from tensorflow.keras import models

def isFile(file):
    return os.path.isfile(file)

if __name__=="__main__":
    # Retrieve given arguments
    dataset, __, word, __ = getDataAndText(sys.argv)

    noise = tf.random.normal([128, 100])
    space = np.zeros((28, 28), np.uint8)

    # Set path to models and backup
    rootpath = "./models/gan_model"
    if dataset != "emnist":
        basepath = os.path.join(rootpath, dataset)
    if not os.path.isdir(basepath):
        sys.exit(dataset, "has no data \n please enter a trained on dataset")
    images = []
    letters = list(word)
    for letter in letters:
        if letter == " " or letter == "_":
            print("Generate image for (space)")
            prediction = space
        else:
            print("Generate image for " + letter)
            # Load generator model corrosponding to letter
            # abdeffghnqrt generally don't differ
            if letter.islower and (letter in "abdefghnqrt") and not letter.isnumeric():
                filename = os.path.join(basepath, "saved_models/g_model_{}_low.h5".format(letter))
                if not isFile(filename):
                    print("missing letter, using emnist models instead")
                    filename = os.path.join(rootpath, "saved_models/g_model_{}_low.h5".format(letter))
            else:
                filename = os.path.join(basepath, "saved_models/g_model_{}.h5".format(letter))
                if not isFile(filename):
                    print("missing letter, using emnist models instead")
                    filename = os.path.join(rootpath, "saved_models/g_model_{}.h5".format(letter))
            model = models.load_model(filename, compile=False)
            # Predict letter with model
            prediction = model(noise)
            # Convert prediction to proper image
            prediction = prediction[1, :, :, 0].numpy()
            prediction = (prediction * 255).astype(np.uint8)
        image = Image.fromarray(prediction, mode="L")
        image = ImageOps.invert(image)
        # Save image to list
        images.append(image)
    
    width = prediction.shape[0]
    height = prediction.shape[1]

    # Merge created images together
    print("Create full image")
    result = Image.new(mode="L", size=(width * len(images), height), color=(255))
    for i in range(len(images)):
        result.paste(images[i], box=(i*width, 0))
    result = result.convert('RGB')
    result.save("./out/{}.png".format(word))
    result.show()
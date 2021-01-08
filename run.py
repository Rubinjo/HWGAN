# Set tensorflow warning level
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from PIL import Image, ImageOps
from cv2 import cv2
import numpy as np

import tensorflow as tf
from tensorflow.keras import models

if __name__=="__main__":
    word = "bad"
    noise = tf.random.normal([128, 100])
    
    images = []
    letters = list(word)
    for letter in letters:
        print("Generate image for " + letter)
        # Load generator model corrosponding to letter
        filename = "./models/gan_model/saved_models/g_model_{}.h5".format(letter)
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
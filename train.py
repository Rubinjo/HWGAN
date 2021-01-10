# Set tensorflow warning level
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import numpy as np
from emnist import extract_training_samples
from emnist import extract_test_samples

import tensorflow as tf
from tensorflow.saved_model import load

from helper.split_data import loaddata
from models.gan_model.GAN import GAN
from models.ocr_model.OCR import OCR

def train_GAN(r_model, images_train, labels_train, images_test, labels_test, characters, character):
    dataset = loaddata(images_train, labels_train, images_test, labels_test, characters, character)
    print("Training GAN...")
    # Create GAN class
    gan = GAN(dataset, character)
    # Create the discriminator
    d_model = gan.define_discriminator()
    # Create the generator
    g_model = gan.define_generator()
    # Train model
    gan.train(g_model, d_model, r_model, characters)
    print("GAN is done")

if __name__=="__main__":
    # Load image data
    full_chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    characters = [char for char in full_chars]

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
    
    # Load EMNIST dataset
    print("\nLoading dataset...")
    images_train, labels_train = extract_training_samples('byclass')
    images_test, labels_test = extract_test_samples('byclass')
    print("Dataset has been loaded")

    # Train OCR model
    print("\nTraining recognizer...")
    ocr = OCR()
    r_model = ocr.define_recognizer()
    ocr.train(r_model, characters, images_train, labels_train, images_test, labels_test)
    print("Recognizer is done")

    # Train GANs for all characters
    for i in range(0, len(characters)):
        character = characters[i]
        print("\nCharacter: " + character)
        train_GAN(r_model, images_train, labels_train, images_test, labels_test, characters, character)
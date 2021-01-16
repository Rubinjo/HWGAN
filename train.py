# Set tensorflow warning level
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import numpy as np
from emnist import extract_training_samples
from emnist import extract_test_samples

import tensorflow as tf
from tensorflow.keras import models

from helper.userinput import *

from helper.conversion import *

from helper.split_data import *
from models.gan_model.GAN import *
from models.ocr_model.OCR import OCR

def train_GAN_EMNIST(r_model, train_images, train_labels, test_images, test_labels, characters, character):
    dataset = loadDataDouble(train_images, train_labels, test_images, test_labels, characters, character)
    print("Training GAN...")
    # Create GAN class
    gan = GAN(dataset, character)
    # Create the discriminator
    d_model = gan.define_discriminator()
    # Create the generator
    g_model = gan.define_generator()
    # Train model
    gan.train(g_model, d_model, r_model, characters)

def train_GAN_USER(folder, r_model, images, labels, char, characters):
    dataset = loadDataSingle(images, labels, char)
    print('Training GAN...')
    # Create GAN class
    gan = GAN(dataset, char)
    # Create the discriminator
    d_model = gan.define_discriminator()
    # Create the generator
    g_model = gan.define_generator()
    # Train model
    # try:
    gan.train(g_model, d_model, r_model, characters, folder)
    # except:
    #     print('an error occured \n Skipping:', char)

if __name__=="__main__":
    # Retrieve given arguments
    dataset, samplesize, splitText, trainOCR = getDataAndText(sys.argv)

    # String of all possible character labels
    full_chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt"
    # Create character list
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
    
    # Check if default dataset is needed
    if trainOCR or dataset == "emnist":
        if samplesize > 0 or splitText != "chars":
            print("\nData is not splitted with the default EMNIST dataset, so your sample and/or text command(s) will be ignored")
        # Load EMNIST dataset
        print("\nLoading EMNIST dataset...")
        images_train, labels_train = extract_training_samples('bymerge')
        images_test, labels_test = extract_test_samples('bymerge')
        print("Dataset has been loaded")
        print("Dataset size:", len(images_train) + len(images_test))

    # Create OCR class
    ocr = OCR()
    # Train OCR model if specified in arguments
    if trainOCR:
        print("\nTraining recognizer...")
        r_model = ocr.define_recognizer(characters)
        print("characters in model:", characters)
        ocr.train(r_model, characters, images_train, labels_train, images_test, labels_test)
        print("Recognizer is done")
    # Load OCR model
    else:
        print("\nLoading recognizer...")
        r_model = ocr.model
        if r_model == None:
            sys.exit("No recognizer is available, please enable training of recognizer")
        print("Recognizer is done")

    # Check if there is a user dataset
    if dataset != "emnist":
        print("\nLoading:", dataset, 'dataset')
        data_chars, data_labels = getDatasetCharLabels(dataset, ocr, characters, splitText)    
        print("Dataset has been loaded")
        print("Dataset size:", len(data_chars))
        if samplesize > 0:
            print("sampling: ", samplesize, "characters")
            try:
                showImages(data_chars[:samplesize], labels = data_labels[:samplesize])
            except:
                showImages(data_chars, labels = data_labels)

        if data_chars != None:
            available_chars = filterDuplicates(data_labels)
            print('available chars in dataset:', available_chars)
            print('creating directory for USER:', dataset)
            try:
                os.mkdir("./models/gan_model/" + dataset)
            except:
                pass
            for char in available_chars:
                print("\nCharacter: " + char)
                train_GAN_USER(dataset, r_model, data_chars, data_labels, char, characters)
    else:
        # Train GANs for all characters
        for i in range(len(characters)):
            character = characters[i]
            print("\nCharacter: " + character)
            train_GAN_EMNIST(r_model, images_train, labels_train, images_test, labels_test, characters, character)
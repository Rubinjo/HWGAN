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
    dataset = loaddata(train_images, train_labels, test_images, test_labels, characters, character)
    print("Training GAN...")
    print('dataset size:', len(dataset))
    # Create GAN class
    gan = GAN(dataset, character)
    # Create the discriminator
    d_model = gan.define_discriminator()
    # Create the generator
    g_model = gan.define_generator()
    # Train model
    gan.train(g_model, d_model, r_model, characters)

def train_GAN_USER(folder, r_model, images, labels, char, characters):
    dataset = getDataset(images, labels, char, targetsize = 1000)
    print('Training GAN...')
    print('dataset size:', len(dataset))
    # Create GAN class
    gan = GAN(dataset, char)
    # Create the discriminator
    d_model = gan.define_discriminator()
    # Create the generator
    g_model = gan.define_generator()
    # Train model
    try:
        gan.train(g_model, d_model, r_model, characters, folder)
    except Exception:
        print('an error occured \n Skipping:', char)

if __name__=="__main__":
    # Get the required dataset (if any)
    dataset, splitLines, samplesize = getDataAndText(sys.argv)
    # Train OCR (or not)
    TRAIN_OCR = False

    # Create possible GAN charcater list
    # full_chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    full_chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt"
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
    if TRAIN_OCR or dataset == "":
        # Load EMNIST dataset
        print("\nLoading dataset...")
        images_train, labels_train = extract_training_samples('bymerge')
        images_test, labels_test = extract_test_samples('bymerge')
        print("Dataset has been loaded")

    # Load OCR model
    ocr = OCR()
    if TRAIN_OCR:
        print("\nTraining recognizer...")
        r_model = ocr.define_recognizer(characters)
        print("characters in model:", characters)
        ocr.train(r_model, characters, images_train, labels_train, images_test, labels_test)
        print("Recognizer is done")
    else:
        print("\nLoading recognizer...")
        r_model = ocr.model
        if r_model == None:
            sys.exit("No recognizer is available, please enable training of recognizer")
        print("Recognizer is done")

    # Check if there is a user dataset
    if dataset != "":
        print("\nLoading:", dataset, 'dataset')
        collectLines = False
        if splitLines == 'split':
            collecLines = True
        data_chars, data_labels = getDatasetCharLabels(dataset, asIndex = False, collectLines = collectLines)
        if samplesize != None:
            try:
                showImages(data_chars[:samplesize], labels = data_labels[:samplesize])
            except Exception:
                showImages(data_chars, labels = data_labels)
                
        if data_chars != None:
            available_chars = filterDuplicates(data_labels)
            print('available chars in dataset:', available_chars)
            print('creating directory for USER:', dataset)
            folder = getGANDir(dataset)
            print('folder:', folder)
            for char in available_chars:
                print("\nCharacter: " + char)
                train_GAN_USER(folder, r_model, data_chars, data_labels, char, characters)
    else:
        # Train GANs for all characters
        #image_set = combineArrays(images_train, images_test)
        #label_set = combineArrays(labels_train, labels_test)
        for i in range(len(characters)):
            character = characters[i]
            print("\nCharacter: " + character)
            train_GAN_EMNIST(r_model, images_train, labels_train, images_test, labels_test, characters, character)
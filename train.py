# Set tensorflow warning level
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import numpy as np
import matplotlib.pyplot as plt
from emnist import extract_training_samples
from emnist import extract_test_samples
import cv2

import tensorflow as tf
from tensorflow.saved_model import load

from helper.split_data import loaddata
from models.gan_model.GAN import GAN
from models.ocr_model.OCR import OCR

#variables
n_gans = []

def load_real_samples(images_train, labels_train, images_test, labels_test, value, value_next, value_previous):
    # load mnist dataset
    training_letter = []
    training_next_letter = []
    training_previours_letter = []

    train_set, test_set = loaddata(images_train, labels_train, images_test, labels_test)
    #all letters in dataset size 28*28
    if value_previous != 100:
        training_previours_letter = train_set[value_previous][0:96]

    if value_next != 100:   
        training_next_letter = train_set[value_next][0:96]
    
    training_letter = train_set[value][:]

    trainX = training_previours_letter + training_letter + training_next_letter
    #trainX = train_set[value][:]
    #trainX, labels_X = extract_training_samples('letters')
    
    #All digits in dataset
    #(trainX, _), (_, _) = load_data()
    
    # expand to 3d, e.g. add channels dimension
    # convert from unsigned ints to floats
    X = np.expand_dims(trainX, axis=-1).astype('float32')
    # scale from [0,255] to [0,1]
    X = X / 255
    return X


def split(word):
    return [char for char in word]

def Dofunction(r_model, images_train, labels_train, images_test, labels_test, value_next, value_previous, value, character):
    dataset = load_real_samples(images_train, labels_train, images_test, labels_test, value, value_next, value_previous)
    print("Training GAN...")
    gan = GAN(dataset, character, NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE, 50)
    # create the discriminator
    d_model = gan.define_discriminator()
    # create the generator
    g_model = gan.define_generator()
    # train model
    gan.train(g_model, d_model, r_model)
    #add gans in list to use a unkonwn number of letters
    n_gans.append(gan)
    print("GAN is done")

if __name__=="__main__":
    # load image data
    Word = "abcdefghijklmnopqrstuvwxyz"
    letters = split(Word)

    #define parameters
    NUM_EPOCHS = 200
    BATCH_SIZE = 128
    LEARNING_RATE = 0.0005

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
    
    alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    alphabet_Capitals = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

    # Load EMNIST dataset
    print("\nLoading dataset...")
    images_train, labels_train = extract_training_samples('letters')
    images_test, labels_test = extract_test_samples('letters')
    print("Dataset has been loaded")

    # Train OCR model
    print("\nTraining recognizer...")
    ocr = OCR()
    r_model = ocr.define_recognizer(alphabet)
    ocr.train(r_model, alphabet, images_train, labels_train, images_test, labels_test)
    print("Recognizer is done")

    for i in range(0, len(letters)):
        next_letter_found = True
        previous_letter_found = True
        letter_found = True
        m,n,p = 0, 0, 0
        value_next = 0
        value_previous = 0
        while(letter_found):
            if(alphabet[p] == letters[i] or alphabet_Capitals[p] == letters[i]):
                letter_found = False
                value = p
            p+=1
            #print("Value of letter: ", value)
        if (i != len(letters)-1):
            next_letter = letters[i+1]
            while(next_letter_found):
                if (alphabet[m] == next_letter or alphabet_Capitals[m] == next_letter):
                    next_letter_found = False
                    value_next = m
                m+=1
            print("\nValue of next letter: ", value_next)
        else:
            value_next = 100


        if (i != 0):
            previous_letter = letters[i-1]
            while(previous_letter_found):
                if (alphabet[n] == previous_letter or alphabet_Capitals[n] == previous_letter):
                    previous_letter_found = False
                    value_previous = n 
                n+=1
            print("Value previous letter: ", value_previous)

        else:
            value_previous = 100 

        if (letters[i] == 'a' or letters[i] == 'A'):
            Dofunction(r_model, images_train, labels_train, images_test, labels_test, value_next, value_previous, value = 0,  character = 'a')
        elif (letters[i] == 'b' or letters[i] == 'B'):
            Dofunction(r_model, images_train, labels_train, images_test, labels_test, value_next, value_previous, value = 1, character = 'b')
        elif (letters[i] == 'c'  or letters[i] == 'Ç'):
            Dofunction(r_model, images_train, labels_train, images_test, labels_test, value_next, value_previous, value = 2, character = 'c')
        elif (letters[i] == 'd'  or letters[i] == 'D'):
            Dofunction(r_model, images_train, labels_train, images_test, labels_test, value_next, value_previous, value = 3, character = 'd')
        elif (letters[i] == 'e'  or letters[i] == 'E'):
            Dofunction(r_model, images_train, labels_train, images_test, labels_test, value_next, value_previous, value = 4, character = 'e')
        elif (letters[i] == 'f'  or letters[i] == 'F'):
            Dofunction(r_model, images_train, labels_train, images_test, labels_test, value_next, value_previous, value = 5, character = 'f')
        elif (letters[i] == 'g'  or letters[i] == 'G'):
            Dofunction(r_model, images_train, labels_train, images_test, labels_test, value_next, value_previous, value = 6, character = 'g')
        elif (letters[i] == 'h'  or letters[i] == 'H'):
            Dofunction(r_model, images_train, labels_train, images_test, labels_test, value_next, value_previous, value = 7, character = 'h')
        elif (letters[i] == 'i'  or letters[i] == 'I'):
            Dofunction(r_model, images_train, labels_train, images_test, labels_test, value_next, value_previous, value = 8, character = 'i')
        elif (letters[i] == 'j'  or letters[i] == 'J'):
            Dofunction(r_model, images_train, labels_train, images_test, labels_test, value_next, value_previous, value = 9, character = 'j')
        elif (letters[i] == 'k'  or letters[i] == 'K'):
            Dofunction(r_model, images_train, labels_train, images_test, labels_test, value_next, value_previous, value = 10, character = 'k')
        elif (letters[i] == 'l'  or letters[i] == 'L'):
            Dofunction(r_model, images_train, labels_train, images_test, labels_test, value_next, value_previous, value = 11, character = 'l')
        elif (letters[i] == 'm'  or letters[i] == 'M'):
            Dofunction(r_model, images_train, labels_train, images_test, labels_test, value_next, value_previous, value = 12, character = 'm')
        elif (letters[i] == 'n'  or letters[i] == 'N'):
            Dofunction(r_model, images_train, labels_train, images_test, labels_test, value_next, value_previous, value = 13, character = 'n')
        elif (letters[i] == 'o'  or letters[i] == 'O'):
            Dofunction(r_model, images_train, labels_train, images_test, labels_test, value_next, value_previous, value = 14, character = 'o')
        elif (letters[i] == 'p'  or letters[i] == 'P'):
            Dofunction(r_model, images_train, labels_train, images_test, labels_test, value_next, value_previous, value = 15, character = 'p')
        elif (letters[i] == 'q'  or letters[i] == 'Q'):
            Dofunction(r_model, images_train, labels_train, images_test, labels_test, value_next, value_previous, value = 16, character = 'q')
        elif (letters[i] == 'r'  or letters[i] == 'R'):
            Dofunction(r_model, images_train, labels_train, images_test, labels_test, value_next, value_previous, value = 17, character = 'r')
        elif (letters[i] == 's'  or letters[i] == 'S'):
            Dofunction(r_model, images_train, labels_train, images_test, labels_test, value_next, value_previous, value = 18, character = 's')
        elif (letters[i] == 't'  or letters[i] == 'T'):
            Dofunction(r_model, images_train, labels_train, images_test, labels_test, value_next, value_previous, value = 19, character = 't')
        elif (letters[i] == 'u'  or letters[i] == 'U'):
            Dofunction(r_model, images_train, labels_train, images_test, labels_test, value_next, value_previous, value = 20, character = 'u')
        elif (letters[i] == 'v'  or letters[i] == 'V'):
            Dofunction(r_model, images_train, labels_train, images_test, labels_test, value_next, value_previous, value = 21, character = 'v')
        elif (letters[i] == 'w'  or letters[i] == 'W'):
            Dofunction(r_model, images_train, labels_train, images_test, labels_test, value_next, value_previous, value = 22, character = 'w')
        elif (letters[i] == 'x'  or letters[i] == 'X'):
            Dofunction(r_model, images_train, labels_train, images_test, labels_test, value_next, value_previous, value = 23, character = 'x')
        elif (letters[i] == 'y'  or letters[i] == 'Y'):
            Dofunction(r_model, images_train, labels_train, images_test, labels_test, value_next, value_previous, value = 24, character = 'y')
        elif (letters[i] == 'z'  or letters[i] == 'Z'):
            Dofunction(r_model, images_train, labels_train, images_test, labels_test, value_next, value_previous, value = 25, character = 'z')
        else:
            print("No of all letters is matched!!")
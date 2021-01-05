# Set tensorflow warning level
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import glob
import time
from pathlib import Path
import sys

from IPython import display
from PIL import Image
import imageio
import matplotlib.pyplot as plt
from cv2 import cv2

import tensorflow as tf
from tensorflow.keras import layers, utils
from tensorflow_docs.vis import embed

# Variables
# Model parameters
EPOCHS = 25
BUFFER_SIZE = 60000
BATCH_SIZE = 256
LEARNING_RATE = 1e-4
# Image dimensions
WIDTH = 28
HEIGHT = 28

noise_dim = 100
num_examples_to_generate = 16
# We will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim])

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

i = 1
# Loop through all directories and files
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
train_images = np.array(images)
# train_labels = np.array(labelsIdx)

# reshape to be [samples][width][height][channels]
train_images = train_images.reshape((train_images.shape[0], WIDTH, HEIGHT, 1)).astype('float32')
# normalize inputs from [0, 255] to [-1, -1]
train_images = (train_images - 127.5) / 127.5
# one hot encode outputs
# train_labels = utils.to_categorical(train_labels)
# Batch and shuffle the data
train_images = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
# train_labels = tf.data.Dataset.from_tensor_slices(train_labels).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
# num_classes = train_labels.shape[1]
# Currently nothing is done with the labels

print("Data has been processed\n")

def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256) # Note: None is the batch size

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    return model

generator = make_generator_model()

def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

discriminator = make_discriminator_model()

recognizer = tf.saved_model.load("./models/ocr_model/")

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def recognizer_loss(rec_real, rec_gen):
    real_loss = cross_entropy(tf.ones_like(rec_real), rec_real)
    fake_loss = cross_entropy(tf.zeros_like(rec_gen), rec_gen)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output, rec_loss):
    gen_loss = cross_entropy(tf.ones_like(fake_output), fake_output)
    total_loss = gen_loss + rec_loss
    return total_loss

generator_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

checkpoint_dir = './models/gan_model/training_checkpoints/'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)
      rec_real = recognizer(images)
      rec_gen = recognizer(generated_images)
      
      # rec_image = recognizer(images)
      real_image = discriminator(images, training=True)
      fake_image = discriminator(generated_images, training=True)
      
      rec_loss = recognizer_loss(rec_real, rec_gen)
      gen_loss = generator_loss(fake_image, rec_loss)
      disc_loss = discriminator_loss(real_image, fake_image)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def train(images, epochs):
  for epoch in range(epochs):
    start = time.time()

    for image_batch in images:
      train_step(image_batch)

    # Produce images for the GIF as we go
    display.clear_output(wait=True)
    generate_and_save_images(generator,
                             epoch + 1,
                             seed)

    # Save the model every 10 epochs
    if (epoch + 1) % 10 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

  # Generate after the final epoch
  display.clear_output(wait=True)
  generate_and_save_images(generator,
                           epochs,
                           seed)

def generate_and_save_images(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = model(test_input, training=False)

  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
      plt.axis('off')

  plt.savefig("./models/gan_model/image_at_epoch_{:04d}.png".format(epoch))

train(train_images, EPOCHS)

#Create gif of all epoch example images
anim_file = "./models/gan_model/gan_process.gif"
with imageio.get_writer(anim_file, mode='I') as writer:
  filenames = glob.glob("./models/gan_model/image_at_epoch*.png")
  filenames = sorted(filenames)
  for filename in filenames:
    image = imageio.imread(filename)
    writer.append_data(image)
  image = imageio.imread(filename)
  writer.append_data(image)
embed.embed_file(anim_file)

#Delete all epoch example images
for f in filenames:
    os.remove(f)
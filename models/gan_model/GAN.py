import os
import numpy as np
from numpy.random import randn, randint
import time
from IPython import display
import matplotlib.pyplot as plt
import imageio
import glob

import tensorflow as tf
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, LeakyReLU, Dropout, BatchNormalization
from tensorflow_docs.vis import embed

class GAN:
    def __init__(self, dataset, character, number_epochs = 100, batch_size = 128, learning_rate = 0.001, r_act_epoch = 50, noise_dim = 100):
        self.dataset = dataset
        self.character = character
        self.N_EPOCHS = number_epochs
        self.N_BATCH = batch_size
        self.LR = learning_rate
        self.R_ACT_EPOCH = r_act_epoch
        self.NOISE = noise_dim
        self.seed = tf.random.normal([16, noise_dim])

    @staticmethod
    def define_discriminator():
        model = Sequential()
        model.add(Conv2D(64, (4,4), strides=(2, 2), padding='same', input_shape=(28,28,1)))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Conv2D(64, (4,4), strides=(2, 2), padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        return model

    # define the standalone generator model
    def define_generator(self):
        model = Sequential()
        # foundation for 7x7 image
        n_nodes = 128 * 7 * 7
        model.add(Dense(n_nodes, input_dim=self.NOISE))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Reshape((7, 7, 128)))
        # upsample to 14x14
        model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        # upsample to 28x28
        model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(1, (7,7), activation='sigmoid', padding='same'))
        return model

    # select real samples
    def generate_real_samples(self, n_samples):
        # choose random instances
        ix = randint(0, self.dataset.shape[0], n_samples)
        # retrieve selected images
        X = self.dataset[ix]
        # generate 'real' class labels (1)
        y = np.ones((n_samples, 1))
        return X, y

    # # generate points in latent space as input for the generator
    # def generate_latent_points(self, latent_dim, n_samples):
    #     # generate points in the latent space
    #     x_input = randn(latent_dim * n_samples)
    #     # reshape into a batch of inputs for the network
    #     x_input = x_input.reshape(n_samples, latent_dim)
    #     return x_input

    # use the generator to generate n fake examples, with class labels
    
    # def generate_fake_samples(self, g_model, latent_dim, n_samples):
    #     # generate points in latent space
    #     x_input = self.generate_latent_points(latent_dim, n_samples)
    #     # predict outputs
    #     X = g_model.predict(x_input)
    #     # create 'fake' class labels (0)
    #     y = np.zeros((n_samples, 1))
    #     return X, y

    def discriminator_loss(self, cross_entropy, real_output, fake_output):
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def recognizer_loss(self, cross_entropy, rec_real, rec_gen):
        real_loss = cross_entropy(tf.ones_like(rec_real), rec_real)
        fake_loss = cross_entropy(tf.zeros_like(rec_gen), rec_gen)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss_with_R(self, cross_entropy, fake_output, rec_loss):
        gen_loss = cross_entropy(tf.ones_like(fake_output), fake_output)
        total_loss = (gen_loss + rec_loss) / 2
        return total_loss
        
    def generator_loss_without_R(self, cross_entropy, fake_output):
        return cross_entropy(tf.ones_like(fake_output), fake_output)

    # Plot performance
    def plot_history(self, d_loss, g_loss):
        # Plot loss
        N = np.arange(0, self.N_EPOCHS)
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(N, d_loss, label="discriminator_loss")
        plt.plot(N, g_loss, label="generator_loss")
        plt.title("Training Loss")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss")
        plt.legend(loc="lower left")
        # save plot image
        filename = "./models/gan_model/graphs/gan_graph_{}.png".format(self.character)
        plt.savefig(filename)
        plt.close()

    def generate_and_save_images(self, model, epoch):
        # Notice `training` is set to False.
        # This is so all layers run in inference mode (batchnorm).
        predictions = model(self.seed, training=False)
        # plot images
        for i in range(predictions.shape[0]):
            # define subplot
            plt.subplot(4, 4, i+1)
            # turn off axis
            plt.axis('off')
            # plot raw pixel data
            plt.imshow(predictions[i, :, :, 0], cmap='gray_r')
        # save plot to file
        plt.savefig("./models/gan_model/gifs/image_at_epoch_{:04d}.png".format(epoch))
        plt.close()
    
    def generate_gif(self):
        anim_file = "./models/gan_model/gifs/gan_process_{}.gif".format(self.character)
        with imageio.get_writer(anim_file, mode='I') as writer:
            filenames = glob.glob("./models/gan_model/gifs/image_at_epoch*.png")
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

    # train the generator and discriminator
    def train(self, g_model, d_model, r_model):
        # Create optimizers
        generator_optimizer = Adam(learning_rate=self.LR)
        discriminator_optimizer = Adam(learning_rate=self.LR)
        # Calculate batch size
        bat_per_epo = int(self.dataset.shape[0] / self.N_BATCH)
        # half_batch = int(self.N_BATCH / 2)
        # new_n_batch = self.N_BATCH

        d_loss_hist, g_loss_hist= list(), list()
        # manually enumerate epochs
        for epoch in range(self.N_EPOCHS):
            start = time.time()
            # enumerate batches over the training set
            for batch in range(bat_per_epo):
                noise = tf.random.normal([self.N_BATCH, self.NOISE])
                # get randomly selected 'real' samples
                X_real, y_real = self.generate_real_samples(batch)
                with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                    # Generate image
                    gen_img = g_model(noise, training=True)
                    
                    # Let discriminator evaluate images
                    real_image = d_model(X_real, training=True)
                    fake_image = d_model(gen_img, training=True)
      
                    # This method returns a helper function to compute cross entropy loss
                    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

                    if epoch > self.R_ACT_EPOCH:
                        # Recoginize images with recognizer
                        rec_real_img = r_model(X_real)
                        rec_gen_img = r_model(gen_img)
                        rec_loss = self.recognizer_loss(cross_entropy, rec_real_img, rec_gen_img)
                        gen_loss = self.generator_loss_with_R(cross_entropy, fake_image, rec_loss)
                    else:    
                        gen_loss = self.generator_loss_without_R(cross_entropy, fake_image)
                    disc_loss = self.discriminator_loss(cross_entropy, real_image, fake_image)

                gradients_of_generator = gen_tape.gradient(gen_loss, g_model.trainable_variables)
                gradients_of_discriminator = disc_tape.gradient(disc_loss, d_model.trainable_variables)

                generator_optimizer.apply_gradients(zip(gradients_of_generator, g_model.trainable_variables))
                discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, d_model.trainable_variables))
            # Create test images
            display.clear_output(wait=True)
            self.generate_and_save_images(g_model, epoch)
            # Save losses of epoch to list
            d_loss_hist.append(disc_loss)
            g_loss_hist.append(gen_loss)
            print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
        # Save model
        g_model.save("./models/gan_model/saved_models/g_model_{}.h5".format(self.character))
        d_model.save("./models/gan_model/saved_models/d_model_{}.h5".format(self.character))
        # Make gif out of test images
        self.generate_gif()
        self.plot_history(d_loss_hist, g_loss_hist)
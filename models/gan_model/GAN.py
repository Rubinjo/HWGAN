import os
import numpy as np
from numpy.random import randn, randint
import time
from IPython import display
import matplotlib.pyplot as plt
import imageio
import glob
from random import randrange

import tensorflow as tf
from tensorflow.keras import optimizers, models, layers
from tensorflow.keras.datasets.mnist import load_data
from tensorflow_docs.vis import embed

class GAN:
    def __init__(self, dataset, character, number_epochs = 128, batch_size = 16, learning_rate = 0.001, r_act_epoch = 16, noise_dim = 100):
        self.dataset = dataset
        self.character = character
        self.N_EPOCHS = number_epochs
        self.N_BATCH = batch_size
        self.LR = learning_rate
        self.R_ACT_EPOCH = r_act_epoch
        self.NOISE = noise_dim
        self.seed = tf.random.normal([16, noise_dim])

    def define_discriminator(self):
        model = models.Sequential()
        # 1st layer - Downsample from 28x28 to 14x14
        model.add(layers.Conv2D(64, (5,5), strides=(2, 2), padding='same', input_shape=(28,28,1), use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))
        # 2nd layer - Downsample to 7x7
        model.add(layers.Conv2D(128, (5,5), strides=(2, 2), padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))
        # 3rd layer
        model.add(layers.Flatten())
        model.add(layers.Dense(1, activation='sigmoid'))
        return model

    # define the standalone generator model
    def define_generator(self):
        model = models.Sequential()
        # 1st layer - Foundation for 7x7 image
        model.add(layers.Dense(7*7*128, use_bias=False, input_dim=self.NOISE))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.Reshape((7, 7, 128)))
        # 2nd layer
        model.add(layers.Conv2DTranspose(64, (5, 5), strides=(1, 1), padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        # 3rd layer - Upsample to 14x14
        model.add(layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        # 4th layer - Upsample to 28x28
        model.add(layers.Conv2DTranspose(16, (5,5), strides=(2,2), padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        # 5th layer
        model.add(layers.Conv2D(1, (7,7), activation='sigmoid', padding='same'))
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

    def recognizer_loss(self, cross_entropy, rec_gen):
        return cross_entropy([1.], [rec_gen])

    def generator_loss_with_R(self, cross_entropy, fake_output, rec_loss):
        gen_loss = cross_entropy(tf.ones_like(fake_output), fake_output)
        total_loss = (gen_loss + rec_loss) / 2
        return total_loss
        
    def generator_loss_without_R(self, cross_entropy, fake_output):
        return cross_entropy(tf.ones_like(fake_output), fake_output)

    # Plot performance
    def plot_history(self, d_loss, g_loss, savedir):
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
        if self.character.isupper() or self.character.isnumeric():
            filename = os.path.join(savedir, "graphs/gan_graph_{}.png".format(self.character))
        else:
            filename = os.path.join(savedir, "graphs/gan_graph_{}_low.png".format(self.character))
        plt.savefig(filename)
        plt.close()

    def generate_and_save_images(self, model, epoch, savedir):
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
        plt.savefig(os.path.join(savedir, "gifs/image_at_epoch_{:04d}.png".format(epoch)))
        plt.close()
    
    def generate_gif(self, savedir):
        if self.character.isupper() or self.character.isnumeric():
            anim_file = os.path.join(savedir, "gifs/gan_process_{}.gif".format(self.character))
        else:
            anim_file = os.path.join(savedir, "gifs/gan_process_{}_low.gif".format(self.character))
        with imageio.get_writer(anim_file, mode='I') as writer:
            filenames = glob.glob(os.path.join(savedir, "gifs/image_at_epoch*.png"))
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
    def train(self, g_model, d_model, r_model, characters, folder = None, makeStats = True):
        # Determine where to save the model
        savedir = "./models/gan_model"
        if folder != None:
            savedir = os.path.join(savedir, folder)

        # Create optimizers
        generator_optimizer = optimizers.Adam(learning_rate=self.LR)
        discriminator_optimizer = optimizers.Adam(learning_rate=self.LR)
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
                    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
                    if epoch >= self.R_ACT_EPOCH:
                        # Recoginize images with recognizer
                        rec_gen_img = r_model(gen_img)
                        gen_pred = rec_gen_img.numpy()
                        gen_pred = gen_pred[randrange(self.N_BATCH)][characters.index(self.character)]

                        rec_loss = self.recognizer_loss(cross_entropy, gen_pred)
                        gen_loss = self.generator_loss_with_R(cross_entropy, fake_image, rec_loss)
                    else:    
                        gen_loss = self.generator_loss_without_R(cross_entropy, fake_image)
                    disc_loss = self.discriminator_loss(cross_entropy, real_image, fake_image)
                gradients_of_generator = gen_tape.gradient(gen_loss, g_model.trainable_variables)
                gradients_of_discriminator = disc_tape.gradient(disc_loss, d_model.trainable_variables)

                generator_optimizer.apply_gradients(zip(gradients_of_generator, g_model.trainable_variables))
                discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, d_model.trainable_variables))
            display.clear_output(wait=True)
            # Create test images
            if makeStats:
                self.generate_and_save_images(g_model, epoch, savedir)
            # Save losses of epoch to list
            if makeStats:
                d_loss_hist.append(disc_loss)
                g_loss_hist.append(gen_loss)
            print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

        # Save model
        if self.character.isupper() or self.character.isnumeric():
            g_model.save(os.path.join(savedir, 'saved_models/g_model_{}.h5'.format(self.character)))
            d_model.save(os.path.join(savedir, 'saved_models/d_model_{}.h5'.format(self.character)))
            #g_model.save("./models/gan_model/saved_models/g_model_{}.h5".format(self.character))
            #d_model.save("./models/gan_model/saved_models/d_model_{}.h5".format(self.character))
        else:
            g_model.save(os.path.join(savedir, 'saved_models/g_model_{}_low.h5'.format(self.character)))
            d_model.save(os.path.join(savedir, 'saved_models/d_model_{}_low.h5'.format(self.character)))
            #g_model.save("./models/gan_model/saved_models/g_model_{}_low.h5".format(self.character))
            #d_model.save("./models/gan_model/saved_models/d_model_{}_low.h5".format(self.character))
        # Make gif out of test images
        if makeStats:
            self.generate_gif(savedir)
            self.plot_history(d_loss_hist, g_loss_hist, savedir)
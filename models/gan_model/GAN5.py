import os
import numpy as np
from numpy.random import randn, randint
import time
from IPython import display
from matplotlib import pyplot
import imageio
import glob

import tensorflow as tf
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, LeakyReLU, Dropout, BatchNormalization
from tensorflow_docs.vis import embed

class GAN:
    def __init__(self, Number_Epochs, Batch_size, Learning_Rate, noise_dim, dataset, character):
        self.n_epochs = Number_Epochs
        self.n_batch = Batch_size
        self.n_lr = Learning_Rate
        self.n_noise = noise_dim
        self.dataset = dataset
        self.character = character
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
        model.add(Dense(n_nodes, input_dim=self.n_noise))
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

    # generate points in latent space as input for the generator
    
    def generate_latent_points(self, latent_dim, n_samples):
        # generate points in the latent space
        x_input = randn(latent_dim * n_samples)
        # reshape into a batch of inputs for the network
        x_input = x_input.reshape(n_samples, latent_dim)
        return x_input

    # use the generator to generate n fake examples, with class labels
    
    def generate_fake_samples(self, g_model, latent_dim, n_samples):
        # generate points in latent space
        x_input = self.generate_latent_points(latent_dim, n_samples)
        # predict outputs
        X = g_model.predict(x_input)
        # create 'fake' class labels (0)
        y = np.zeros((n_samples, 1))
        return X, y

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
        print(gen_loss)
        total_loss = gen_loss + rec_loss
        return total_loss
        
    def generator_loss_without_R(self, cross_entropy, fake_output):
        return cross_entropy(tf.ones_like(fake_output), fake_output)

    # create and save a plot of generated images (reversed grayscale)
    def save_plot(self, examples, epoch, n=1):
        # plot images
        for i in range(n * n):
            # define subplot
            pyplot.subplot(n, n, 1 + i)
            # turn off axis
            pyplot.axis('off')
            # plot raw pixel data
            pyplot.imshow(examples[i, :, :, 0], cmap='gray_r')
        # save plot to file
        filename = 'out/generated_plot_%s.png' % (self.character)
        pyplot.savefig(filename)
        pyplot.close()

    # evaluate the discriminator, plot generated images, save generator model
    def summarize_performance(self, epoch, g_model, d_model, latent_dim, n_samples=1):
        # prepare real samples
        X_real, y_real = self.generate_real_samples(n_samples)
        # evaluate discriminator on real examples
        _, acc_real = d_model.evaluate(X_real, y_real, verbose=0)
        # prepare fake examples
        x_fake, y_fake = self.generate_fake_samples(g_model, latent_dim, n_samples)
        # evaluate discriminator on fake examples
        _, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
        # summarize discriminator performance
        print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real*100, acc_fake*100))
        # save plot
        self.save_plot(x_fake, epoch)
        # save the generator model tile file
        filename = 'Generatedmodel/generator_model_%03d.h5' % (epoch + 1)
        g_model.save(filename)

    #plot performance
    def plot_history(self, d1_hist, d2_hist, g_hist, a1_hist, a2_hist):
        # plot loss
        pyplot.subplot(2, 1, 1)
        pyplot.plot(d1_hist, label='discriminator-real')
        pyplot.plot(d2_hist, label='discriminator-fake')
        pyplot.plot(g_hist, label='gen')
        pyplot.legend()
        # plot discriminator accuracy
        pyplot.subplot(2, 1, 2)
        pyplot.plot(a1_hist, label='acc_disc-real')
        pyplot.plot(a2_hist, label='acc-disc-fake')
        pyplot.legend()

        # save plot to file
        filename_plt = 'performance_results/plot_line_plot_loss_%s.png' % (self.character)
        pyplot.savefig(filename_plt)
        
        pyplot.close()

    def generate_and_save_images(self, model, epoch):
        # Notice `training` is set to False.
        # This is so all layers run in inference mode (batchnorm).
        predictions = model(self.seed, training=False)
        # plot images
        for i in range(predictions.shape[0]):
            # define subplot
            pyplot.subplot(4, 4, i+1)
            # turn off axis
            pyplot.axis('off')
            # plot raw pixel data
            pyplot.imshow(predictions[i, :, :, 0], cmap='gray_r')
        # save plot to file
        pyplot.savefig("./models/gan_model/image_at_epoch_{:04d}.png".format(epoch))
        pyplot.close()
    
    def generate_gif(self):
        anim_file = "./models/gan_model/gan_process_{}.gif".format(self.character)
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

    # train the generator and discriminator
    def train(self, g_model, d_model, r_model):
        # Create optimizers
        generator_optimizer = Adam(learning_rate=self.n_lr)
        discriminator_optimizer = Adam(learning_rate=self.n_lr)
        # Calculate batch size
        bat_per_epo = int(self.dataset.shape[0] / self.n_batch)
        # half_batch = int(self.n_batch / 2)
        # new_n_batch = self.n_batch

        # disc1_hist, disc2_hist, gan_hist, acc1_hist, acc2_hist = list(), list(), list(), list(), list()
        # manually enumerate epochs
        for epoch in range(self.n_epochs):
            start = time.time()
            # enumerate batches over the training set
            for batch in range(bat_per_epo):
                noise = tf.random.normal([self.n_batch, self.n_noise])
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

                    if epoch > 50:
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
            display.clear_output(wait=True)
            self.generate_and_save_images(g_model, epoch)
            print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
        display.clear_output(wait=True)
        self.generate_and_save_images(g_model, self.n_epochs)
        self.generate_gif()
                # summarize loss on this batch
                # record history
                # disc1_hist.append(disc_loss)
                # disc2_hist.append(disc_loss2)
                # gan_hist.append(gan_loss)
                # acc1_hist.append(disc_acc1)
                # acc2_hist.append(disc_acc2)
                #print('>%d, %d/%d, d=%.3f, g=%.3f' % (i+1, j+1, bat_per_epo, d_loss, g_loss)) I have comment thisone to speedup the cyclus
                
                # evaluate the model performance, sometimes
            # evaluate the model performance, sometimes
            
            #if (i+1) % 1 == 0:
                #self.summarize_performance(i, g_model, d_model, latent_dim)
        # self.plot_history(disc1_hist, disc2_hist, gan_hist, acc1_hist, acc2_hist)
        # self.summarize_performance(i, g_model, d_model, latent_dim)
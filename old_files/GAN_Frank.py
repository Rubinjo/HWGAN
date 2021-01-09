import numpy as np
from numpy.random import randn, randint
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, LeakyReLU, Dropout, BatchNormalization
from matplotlib import pyplot

class GAN:
    def __init__(self, Number_Epochs, Batch_size, dataset, character):
        self.n_epochs = Number_Epochs
        self.n_batch = Batch_size
        self.dataset = dataset
        self.character = character

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
        # compile model
        opt = Adam(lr=0.0004, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        return model

    # define the standalone generator model
    @staticmethod
    def define_generator(latent_dim):
        model = Sequential()
        # foundation for 7x7 image
        n_nodes = 128 * 7 * 7
        model.add(Dense(n_nodes, input_dim=latent_dim))
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

    # define the combined generator and discriminator model, for updating the generator
    
    def define_gan(self, g_model, d_model):
        # make weights in the discriminator not trainable
        d_model.trainable = False
        # connect them
        model = Sequential()
        # add generator
        model.add(g_model)
        # add the discriminator
        model.add(d_model)
        # compile model
        opt = Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt)
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

    # train the generator and discriminator
    def train(self, g_model, d_model, gan_model, latent_dim):
        bat_per_epo = int(self.dataset.shape[0] / self.n_batch)
        half_batch = int(self.n_batch / 2)
        new_n_batch = self.n_batch
        disc1_hist, disc2_hist, gan_hist, acc1_hist, acc2_hist = list(), list(), list(), list(), list()
        # manually enumerate epochs
        for i in range(self.n_epochs):
            # enumerate batches over the training set
            for j in range(bat_per_epo):
                # get randomly selected 'real' samples
                X_real, y_real = self.generate_real_samples(half_batch)
                disc_loss1, disc_acc1 = d_model.train_on_batch(X_real, y_real)
                # generate 'fake' examples
                X_fake, y_fake = self.generate_fake_samples(g_model, latent_dim, half_batch)
                disc_loss2, disc_acc2 = d_model.train_on_batch(X_fake, y_fake)
                # create training set for the discriminator
                #X, y = np.vstack((X_real, X_fake)), np.vstack((y_real, y_fake))
                
                # update discriminator model weights
                #d_loss, d_acc = d_model.train_on_batch(X, y)
                
                # prepare points in latent space as input for the generator
                X_gan = self.generate_latent_points(latent_dim, new_n_batch)
                # create inverted labels for the fake samples
                y_gan = np.ones((self.n_batch, 1))
                # update the generator via the discriminator's error
                gan_loss = gan_model.train_on_batch(X_gan, y_gan)
                # summarize loss on this batch
                # record history
                disc1_hist.append(disc_loss1)
                disc2_hist.append(disc_loss2)
                gan_hist.append(gan_loss)
                acc1_hist.append(disc_acc1)
                acc2_hist.append(disc_acc2)
                #print('>%d, %d/%d, d=%.3f, g=%.3f' % (i+1, j+1, bat_per_epo, d_loss, g_loss)) I have comment thisone to speedup the cyclus
                
                # evaluate the model performance, sometimes
            # evaluate the model performance, sometimes
            
            #if (i+1) % 1 == 0:
                #self.summarize_performance(i, g_model, d_model, latent_dim)
        self.plot_history(disc1_hist, disc2_hist, gan_hist, acc1_hist, acc2_hist)
        self.summarize_performance(i, g_model, d_model, latent_dim)
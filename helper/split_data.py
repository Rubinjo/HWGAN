import numpy as np

# Load specific char dataset
def loaddata(images_train, labels_train, images_test, labels_test, characters, character):
    images = []
    # Load all images of dataset that correspond to character
    for i in range(0, len(labels_train)):
        if (characters[labels_train[i]] == character):
            images.append(images_train[i])
    for i in range(0, len(labels_test)):
        if (characters[labels_test[i]] == character):
            images.append(images_test[i])

    # reshape to be [samples][width][height][channels]
    images = np.expand_dims(images, axis=-1).astype('float32')
    # normalize inputs from 0-255 to 0-1
    images = images / 255
    return images
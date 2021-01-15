import numpy as np

def grays_to_float32(imgs, normalized = True):
    im_n = np.expand_dims(imgs, axis=1).astype('float32')
    im_n = np.expand_dims(im_n, axis=-1)
    if normalized:
        return im_n / 255
    return im_n
    
# Load specific char dataset
def loadDataDouble(train_images, train_labels, test_images, test_labels, characters, character):
    images = []
    # Load all images of dataset that correspond to character
    for i in range(len(train_labels)):
        if (characters[train_labels[i]] == character):
            images.append(train_images[i])
    for i in range(len(test_labels)):
        if (characters[test_labels[i]] == character):
            images.append(test_images[i])
    # reshape to be [samples][width][height][channels]
    images = np.expand_dims(images, axis=-1).astype('float32')
    # normalize inputs from 0-255 to 0-1
    images = images / 255
    return images


def loadDataSingle(images, labels, char):
    data_images = []
    # Load all images of dataset that correspond to character
    for i in range(len(labels)):
        if (labels[i] == char):
            data_images.append(images[i])
    # reshape to be [samples][width][height][channels]
    data_images = np.expand_dims(data_images, axis=-1).astype('float32')
    # normalize inputs from 0-255 to 0-1
    data_images = data_images / 255
    return data_images
    
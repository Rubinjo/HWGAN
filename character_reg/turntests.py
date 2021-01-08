import sys
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as im
import cv2 as cv
from scipy.ndimage import interpolation as inter


def simplify(img, tresh = 200, invert = False):
    h,w = img.shape[:2]
    out = np.zeros([h,w])
    for y in range(h):
        for x in range(w):
            if img[y][x] < tresh:
                out[y][x] = 0
            else:
                out[y][x] = 1
    if invert:
        return (1 - out)
    else:
        return out

input_file = sys.argv[1]


# returns the total amount of difference between 1 - 0 on the x axis
# a high score the maximum amount of difference
def find_score(arr, angle):
    data = inter.rotate(arr, angle, reshape=False, order=0)
    hist = np.sum(data, axis=1)
    score = np.sum((hist[1:] - hist[:-1]) ** 2)
    return hist, score

# rotates the image to put the lines straight, based on the total amount of black pixels
# method adapted from https://towardsdatascience.com/pre-processing-in-ocr-fc231c6035a7
def straightenImage(binary, asCV = True):
    h,w = binary.shape[:2]
    print(w, h)
    flipped = cv.transpose(binary)
    flipped = cv.flip(flipped, flipCode=0)
    d = 0.5
    lim = 10
    angles = np.arange(-lim, lim + d, d)
    scores = []
    #try to rotate the image a +- 5 degrees with 1 degree increments
    for angle in angles:
        hist, score = find_score(binary, angle)
        scores.append(score)
    flippedscores = []
    for angle in angles:
        hist, score = find_score(flipped, angle)
        flippedscores.append(score)

    best_normal = max(scores)
    best_flipped = max(flippedscores)
    print(best_flipped, best_normal)
    if best_normal > best_flipped:
        print('normal is better')
        best_angle = angles[scores.index(best_normal)]
        print(best_angle)
        corrected = inter.rotate(binary, best_angle, reshape = False, order = 0)
        return im.fromarray((255 * corrected).astype("uint8")).convert("RGB")
    else:
        print('flipped is better')
        best_angle = angles[flippedscores.index(best_flipped)]
        print(best_angle)
        corrected = inter.rotate(flipped, best_angle, reshape = False, order = 0)
        return im.fromarray((255 * corrected).astype("uint8")).convert("RGB")
    if asCV:
        return corrected
    else:
        return im.fromarray((255 * corrected).astype("uint8")).convert("RGB")


img = cv.imread(input_file, cv.IMREAD_GRAYSCALE)
bin_img = simplify(img, tresh = 40, invert = True)
transposed = straightenImage(bin_img, asCV = False)
transposed.save('skew_corrected.png')
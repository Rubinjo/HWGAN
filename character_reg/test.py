import sys
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as im
from scipy.ndimage import interpolation as inter
import cv2 as cv
import math
import io

from conversion import *

#from models.ocr_model.OCR import OCR

def bin_to_img(bin):
    return im.fromarray((255 * bin).astype("uint8")).convert("RGB")

gray = cv.imread('pen.jpg', cv.IMREAD_GRAYSCALE)

binary = convSimplify(gray, k_size = 10, invert = True)
lines, graphs = splitLines(binary)

characters = []
for line in lines:
    chars = splitChars(line)
    for char in chars:
        c = squareChar(char)
        characters.append(c)
characters, noncharacters = filterOutliers(characters)
print(len(characters))

#ocr = OCR()

characters = resizeImages(characters, size = 28)
characters = bins_to_float32(characters)
labels = []
for char in characters:
    #label = ocr.predict(char)
    labels.append('a')
showImages(characters, labels = labels)


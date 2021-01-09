import sys
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as im
from scipy.ndimage import interpolation as inter
import cv2 as cv
import math
import io

import os
from pathlib import Path

from conversion import *

#for debugging:
graphs = []

#denoising is useful since during the rgb -> bin process noise is likely to get introduced
DENOISING = False
a = 50
b = 15
c = 7
d = 15

#erosion is the thinning of lines (OPTIONAL)
EROSION = False
#kernel for erosion: (how many pixels to remove)
kernel = np.ones((2,2),np.uint8)

# Dataset folder
rootdir = Path("./input")
outdir = Path("./output")

def safeData(name, data):
    folder = os.path.join(outdir, name)
    if os.path.isdir(folder) == False:
        os.mkdir(folder)
    savepath = os.path.join(folder,  'skewed' + '.png')
    print('saving skewed:', savepath)

def np_to_img(img, binary = False):
    if binary:
        return im.fromarray((255 * img).astype("uint8")).convert("RGB")
    else:
        return im.fromarray(img).astype("uint8t").convert("RGB") 

def erodeImage(img, times = 1):
    return cv.erode(img, kernel, iterations = times)

def smoothImage(img):
    return cv.fastNlMeansDenoisingColored(img, None, a, b, c, d)

def processImage(name, binary):
    h, w = binary.shape[:2]
    print(name, 'height:', h, 'width', w)
    post = straightenImage(binary)
    lines = splitLines(post, graphs)
    for line in lines:
        l = np_to_img(line, binary = True)
        l = smoothImage(l)

#pre processing:
for root, dirs, files in os.walk(rootdir):
    for name in files:
        path = os.path.join(root, name)
        print('loading: ', path)
        try:
            no_ext = os.path.splitext(name)[0]
            gray = cv.imread(path, cv.IMREAD_GRAYSCALE)
            binary = simplify(gray, tresh = 40, invert = True)
            processImage(no_ext, binary)
        except Exception:
            print('file: ' + path + " is not usable")
            continue

print(len(graphs))
for i in range(len(graphs)):
    img = im.fromarray((255 * graphs[i]).astype("uint8")).convert("RGB")
    img.save('graph' + str(i) + '.png')

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

#defines for post processing

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

def PIL_to_CV(img):
    open_cv_image = np.array(img)
    #RGB to BGR 
    return open_cv_image[:, :, ::-1].copy()

#binary array to cv2_gray
def expand(img):
    h,w = img.shape[:2]
    out = np.zeros([h,w], np.uint8)
    for y in range(h):
        for x in range(w):
            if img[y][x] == 1:
                out[y][x] = 255
    return out

def invert(img, max = 1):
    return (max - img)

def showImages(imgs):
    w=10
    h=10
    fig = plt.figure(figsize=(8, 8))
    columns = 4
    rows = math.ceil(len(imgs) / columns)
    print(len(imgs), 'cols = ', columns, 'rows =', rows)
    for i in range(1, len(imgs) + 1):
        fig.add_subplot(rows, columns, i)
        plt.imshow(imgs[i - 1])
    plt.show()

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
    d = 1
    lim = 5
    angles = np.arange(-lim, lim + d, d)
    scores = []
    #try to rotate the image a +- 5 degrees with 1 degree increments
    for angle in angles:
        hist, score = find_score(binary, angle)
        scores.append(score)
    best_score = max(scores)
    best_angle = angles[scores.index(best_score)]
    corrected = inter.rotate(binary, best_angle, reshape = False, order = 0)
    if asCV:
        return corrected
    else:
        return im.fromarray((255 * corrected).astype("uint8")).convert("RGB")

def conv1d(scores, index, r = 10):
    x1 = index - r
    x2 = index + r
    if x1 < 0:
        x1 = 0
    if x2 > len(scores):
        x2 = len(scores)
    score = np.sum(scores[x1:x2])
    return score / (x2 - x1)

def findLowStreaks(scores, tresh = 0):
    streaks = []
    streakInProgress = False
    streak = []
    for i in range(len(scores)):
        if scores[i] <= tresh:
            if streakInProgress == False:
                streakInProgress = True
            streak.append(i)
        else:
            if streakInProgress:
                streaks.append(streak)
                streak = []
                streakInProgress = False
    if streakInProgress:
        streaks.append(streak)
    return streaks 

def splitLines(binary):
    hist = np.sum(binary, axis=1)
    streaks = findLowStreaks(hist)
    splits = []
    for streak in streaks:
        avg = int(round(np.sum(streak) / len(streak)))
        splits.append(avg)
    lines = []
    for i in range(len(splits) - 1):
        p1 = splits[i]
        p2 = splits[i + 1]
        line = binary[p1:p2][:]
        lines.append(line)
    return lines

def getBounds(cnt):
    minx = 100000
    maxx = 0
    miny = 100000
    maxy = 0
    for c in cnt:
        point = c[0]
        x = point[0]
        y = point[1]
        if x < minx:
            minx = x
        if x > maxx:
            maxx = x
        if y < miny:
            miny = y
        if y > maxy:
            maxy = y
    return [(maxy - miny) + 1, (maxx - minx) + 1, minx, miny, maxx, maxy]

def getContours(gray):
    ret, thresh = cv.threshold(gray, 127, 255, 0)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    return contours

def cnt_to_imgs(contours):
    imgs = []
    for cnt in contours:
        b = getBounds(cnt)
        img = np.zeros(b[:2])
        for c in cnt:
            point = c[0] - b[2:4]
            img[point[1]][point[0]] = 255
        imgs.append(img)
    return imgs

def groupCharacters(line):
    gray = expand(line)
    #showImages([line, gray])
    contours = getContours(gray)
    bounds = cnt_to_imgs(contours)
    #print(len(bounds))
    #showImages(bounds)
    #print(len(contours))
    dsp = bounds + [line]
    showImages(dsp)
    hist = np.sum(line, axis = 0)
    x_axis = np.arange(0, len(hist), 1)
    

    #plt.show()

#pre processing:
for root, dirs, files in os.walk(rootdir):
    for name in files:
        path = os.path.join(root, name)
        print(path)
        try:
            no_ext = os.path.splitext(name)[0]
            gray = cv.imread(path, cv.IMREAD_GRAYSCALE)
            binary = simplify(gray, tresh = 40, invert = True)
            post = straightenImage(binary)
            lines = splitLines(post)
            print(no_ext, 'has', len(lines), 'lines')
            folder = os.path.join(outdir, no_ext)
            os.mkdir(folder) 
            #groupCharacters(lines[0])
            for i in range(len(lines)):
                ID = 'line_' + str(i) + '.png'
                outpath = os.path.join(outdir, no_ext, ID)
                img = im.fromarray((255 * lines[i]).astype("uint8")).convert("RGB")
                img.save(outpath)
            #post = PIL_to_CV(post)
            if DENOISING:
                post = cv.fastNlMeansDenoisingColored(post, None, a, b, c, d)
            if EROSION:
                post = cv.erode(post, kernel,iterations = 1)

        except Exception:
            print('file: ' + path + " is not usable")
            continue


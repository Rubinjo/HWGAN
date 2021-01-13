import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as im
from scipy.ndimage import interpolation as inter
import cv2 as cv
import math
import io

from models.ocr_model.OCR import OCR

import os
from pathlib import Path

def combineArrays(arr1, arr2):
    out = []
    for a in arr1:
        out.append(a)
    for a in arr2:
        out.append(a)
    return out

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

def rangeExtremes(gray, c1, c2):
    minimum = 255
    maximum = 0
    for y in range(c1[0], c2[0]):
        for x in range(c1[1], c2[1]):
            if gray[y][x] > maximum:
                maximum = gray[y][x]
            if gray[y][x] < minimum:
                minimum = gray[y][x]
    return minimum, maximum

def convSimplify(gray, div = 10, k_size = 10, mindiff = 70, invert = False):
    h, w = gray.shape[:2]
    binary = np.ones((h,w))

    kernel = np.ones((k_size,k_size),np.float32)/ (k_size ** 2)
    smoothed = cv.filter2D(gray,-1,kernel)
    xstep = int(math.ceil(w / div))
    ystep = int(math.ceil(h / div))

    # print(xstep, ystep)
    x = 0
    y = 0
    for y1 in range(0, h, ystep):
        y2 = y1 + ystep
        if y2 > h:
            y2 = h
        for x1 in range(0, w, xstep):
            x2 = x1 + xstep
            if x2 > w:
                x2 = w
            minimum, maximum = rangeExtremes(smoothed, (y1, x1), (y2, x2))
            if maximum - minimum > mindiff:
                split = 60
                #split = int((maximum - minimum) / 2)
                for y in range(y1, y2):
                    for x in range(x1, x2):
                        if gray[y][x] < split:
                            binary[y][x] = 0
                        else:
                            binary[y][x] = 1
    if invert:
        return (1 - binary)
    else:
        return binary

def PIL_to_CV(img):
    open_cv_image = np.array(img)
    #RGB to BGR 
    return open_cv_image[:, :, ::-1].copy()

def np_to_img(img, binary = False):
    if binary:
        return im.fromarray((255 * img).astype("uint8")).convert("RGB")
    else:
        return im.fromarray(img).astype("uint8t").convert("RGB") 

def make_tf_compatible(img):
    img = img.astype('float32')
    imgs = np.expand_dims(img, axis=1)
    
def grays_to_float32(imgs, normalized = True):
    im_n = np.expand_dims(imgs, axis=1).astype('float32')
    im_n = np.expand_dims(im_n, axis=-1)
    if normalized:
        return im_n / 255
    return im_n

#binary array to cv2_gray
def expand(img):
    h,w = img.shape[:2]
    out = np.zeros([h,w], np.uint8)
    for y in range(h):
        for x in range(w):
            if img[y][x] == 1:
                out[y][x] = 255
    return out

def norms_to_grays(imgs):
    out = []
    for img in imgs:
        h,w = img.shape[:2]
        new = np.zeros((h,w),np.uint8)
        for y in range(h):
            for x in range(w):
                new[y][x] = int(img[y][x] * 255)
        out.append(new)
    return out
    #for img in imgs:
        
def invert(img, max = 1):
    return (max - img)

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

def showImages(imgs, labels = None, columns = False):
    fig = plt.figure(figsize=(8, 8))
    if columns == False:
        columns = int(math.sqrt(len(imgs)))
    rows = math.ceil(len(imgs) / columns)
    print(len(imgs), 'cols = ', columns, 'rows =', rows)
    for i in range(1, len(imgs) + 1):
        fig.add_subplot(rows, columns, i)
        plt.imshow(imgs[i - 1])
        if (labels != None):
            plt.title(labels[i - 1])
    plt.show()

def convline(scores, index, r = 10):
    x1 = index - r
    x2 = index + r
    if x1 < 0:
        x1 = 0
    if x2 > len(scores):
        x2 = len(scores)
    score = np.sum(scores[x1:x2])
    return score / (x2 - x1)

def conv1d(scores, r = 10):
    out = []
    for i in range(len(scores)):
        out.append(convline(scores, i, r))
    return out

def convSmooth(scores, cycles = 1, area = 10):
    avg = scores
    for i in range(cycles):
        avg = conv1d(avg, r = area)
    return avg

def histToImage(hist):
    h = len(hist)
    w = int(math.ceil(max(hist)))
    box = np.zeros((h,w), np.uint8)
    for i in range(h):
        val = int(hist[i])
        box[i][:val] = 1
    return box

def extreme_in_range(scores, x, tresh = 1, r = 10, findmax = True):
    x1 = x - r
    x2 = x + r
    if x1 < 0:
        x1 = 0
    if x2 > len(scores) - 1:
        x2 = len(scores) - 1
    sample = scores[x1:x2]
    localavg = np.sum(sample) / (x2 - x1)
    if findmax:
        if scores[x] == max(sample) and scores[x] >= localavg + tresh:
            return True
        else:
            return False
    else:
        if scores[x] == min(sample) and scores[x] <= localavg - tresh:
            return True
        else:
            return False

def findLocalMaxima(scores, r = 10):
    maxima = []
    for i in range(1, len(scores) - 1):
        if extreme_in_range(scores, i):
            maxima.append(i)
    return maxima

def getAvgSpacing(arr):
    jumps = 0
    for i in range(len(arr) - 1):
        jumps += (arr[i + 1] - arr[i])
    return jumps / len(arr)

def findBounds(maxima, scores, area = 10):
    bounds = []
    # print(maxima)
    for maximum in maxima:
        bound = []
        found = False
        #find a valley backwards
        for i in reversed(range(0, maximum)):
            if extreme_in_range(scores, i, r = area, tresh = 0, findmax = False):
                bound.append(i)
                found = True
                break
        if found == False:
            bound.append(0)
        found = False
        #find a valley forwards
        for i in range(maximum, len(scores)):
            if extreme_in_range(scores, i, r = area, tresh = 0, findmax = False):
                bound.append(i)
                found = True
                break
        if found == False:
            bound.append(len(scores))
        bounds.append(bound)
    return bounds
    
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

def streakToBound(streaks):
    bounds = []
    for i in range(len(streaks) - 1):
        bound = []
        s1 = streaks[i]
        s2 = streaks[i + 1]
        bound.append(s1[len(s1) - 1])
        bound.append(s2[0])
        bounds.append(bound)
    return bounds

def inList(e, arr):
    for a in arr:
        if e == a:
            return True
    return False

def filterDuplicates(arr):
    out = []
    for a in arr:
        if not inList(a, out):
            out.append(a)
    return out

def splitLines(binary):
    graphs = []
    hist = np.sum(binary, axis=1)
    avgs = convSmooth(hist, 5)
    graphs.append(histToImage(avgs))
    maxima = findLocalMaxima(avgs, r = 30)
    print(maxima)
    print(avgs)
    projectedSpacing = int(getAvgSpacing(maxima) / 2)
    print(projectedSpacing)
    print("here1")
    bounds = findBounds(maxima, avgs, area = projectedSpacing)
    print("here2")
    bounds = filterDuplicates(bounds)

    lines = []
    for bound in bounds:
        line = binary[bound[0]:bound[1]][:]
        lines.append(line)
    return lines, graphs

def saveNPGraphs(graphs):
    for i in range(len(graphs)):
        img = im.fromarray((255 * graphs[i]).astype("uint8")).convert("RGB")
        img.save('graph' + str(i) + '.png')

def findStart(hist):
    for i in range(len(hist)):
        if hist[i] != 0:
            return i

def findEnd(hist):
    for i in reversed(range(len(hist))):
        if hist[i] != 0:
           return i

def squareChar(char):
    h, w = char.shape[:2]
    if w > h:
        diff = w - h
        u = int(diff / 2)
        #print('w,h', w, h, 'l', u, u + h)
        out = np.zeros((w,w))
        for y in range(h):
            for x in range(w):
                out[y + u][x] = char[y][x]
        return out
    elif h > w:
        diff = h - w
        l = int(diff / 2)
        #print('w,h', w, h, 'l', l, l + w)
        out = np.zeros((h,h))
        for y in range(h):
            for x in range(w):
                out[y][x + l] = char[y][x]
        return out
    else: 
        return char

def resizeImages(imgs, size = 96):
    out = []
    for img in imgs:
        out.append(cv.resize(img, (size,size), interpolation = cv.INTER_AREA))
    return out

#assumes a square input
def filterOutliers(imgs, stdevs = 1):
    sizes = []
    for img in imgs:
        h,w = img.shape[:2]
        sizes.append(h)
    std_ = np.std(sizes)
    mean_ = np.mean(sizes)
    norm = []
    outlier = []
    for img in imgs:
        h,w = img.shape[:2]
        if h > mean_ - (std_ * stdevs) and h < mean_ + (std_ * stdevs):
            norm.append(img)
        else:
            outlier.append(img)
    return norm, outlier

def cleanChar(char):
    hist = np.sum(char, axis = 1)
    start = findStart(hist)
    end = findEnd(hist)
    return char[start:end][:]

def cleanChars(chars):
    out = []
    for char in chars:
        out.append(cleanChar(char))
    return out

def seperateChars(bounds, line):
    chars = []
    h, w = line.shape[:2]
    for bound in bounds:
        char = np.zeros((h, bound[1] - bound[0]))
        for x in range(bound[0], bound[1]):
            for y in range(h):
                char[y][x - bound[0]] = line[y][x]
        char = cleanChar(char)
        chars.append(char)
    return chars

def splitChars(line, graphs = []):
    hist = np.sum(line, axis=0)
    avgs = convSmooth(hist, 1, area = 4)
    graphs.append(histToImage(avgs))
    streaks = findLowStreaks(avgs)
    bounds = streakToBound(streaks)
    chars = seperateChars(bounds, line)
    #showImages(chars)
    # print(chars[0].shape[:2])
    return chars

def getCharactersWithLabels(path, asIndex = True, ocr = None,):
    gray = cv.imread(path, cv.IMREAD_GRAYSCALE)
    binary = convSimplify(gray, k_size = 10, invert = True)
    lines, graphs = splitLines(binary)
    characters = []
    line_graphs = []
    for line in lines:
        chars = splitChars(line, graphs = line_graphs)
        for char in chars:
            c = squareChar(char)
            characters.append(c)
    characters, noncharacters = filterOutliers(characters)
    if ocr == None:
        ocr = OCR()

    # showImages(line_graphs[:-1], columns = len(line_graphs))
    characters = resizeImages(characters, size = ocr.WIDTH)
    characters = norms_to_grays(characters)

    tf_chars = grays_to_float32(characters)

    fullAlphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    all_labels = [char for char in fullAlphabet]

    labels = []
    for char in tf_chars:
        index = np.argmax(ocr.model(char))
        if asIndex:
            labels.append(index)
        else:
            labels.append(all_labels[index])
    
    return characters, labels

def getUserCharLabels(user, asIndex = True):
    rootdir = Path("./dataset")
    characters = []
    labels = []
    userpath = os.path.join(rootdir, user)
    if not os.path.isdir(userpath):
        return None, None
    ocr = OCR()
    for root, dirs, files in os.walk(userpath):
        for name in files:
            path = os.path.join(root, name)
            print('extracting characters from:', path)
            try:
                chars, labs = getCharactersWithLabels(path, asIndex = asIndex, ocr = ocr)
                characters += chars
                labels += labs
            except Exception:
                print('file: ', path, 'is not usable and will be skipped')
                continue
    return characters, labels


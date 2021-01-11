# import sys
# import matplotlib.pyplot as plt
# import numpy as np
# from PIL import Image as im
# from scipy.ndimage import interpolation as inter
# import cv2 as cv
# import math
# import io

# from conversion import *

# from emnist import extract_test_samples

#characters, labels = getUserCharLabels('tom', asIndex = False)

#showImages(characters, labels = labels)

txt = "-user tom -text hello"

def getArg(arg):
    if arg[0] == '-':
        identifier = arg[1:]
        if identifier == 'user':
            return True, 'user'
        elif identifier == 'text':
            return True, 'text'
    else:
        return False, 'none'

def getUserAndText(args):
    textParsing = False
    expectingUser = False
    user = 'emnist'
    text = ""
    for arg in args:
        isarg, sort = getArg(arg)
        if isarg:
            if sort == 'user':
                expectingUser = True
                textParsing = False
                continue
            elif sort == "text":
                textParsing = True
                expectingUser = False
                continue
        if expectingUser:
            user = arg
            expectingUser = False
            continue
        if textParsing:
            text += arg
            text += " "
    if text != "":
        text = text[:len(text) - 1]
    return user, text

user, text = getUserAndText(args)

print(user, text)

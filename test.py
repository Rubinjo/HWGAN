import sys
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as im
from scipy.ndimage import interpolation as inter
import cv2 as cv
import math
import io

from conversion import *

from emnist import extract_test_samples

characters, labels = getUserCharLabels('tom', asIndex = False)

showImages(characters, labels = labels)


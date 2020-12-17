from cv2 import cv2
import numpy as np
import imutils
from imutils.contours import sort_contours

"""
Calculate how many non-whitish pixels are arround the currently selected pixel
"""
def neighborScore(idxPixelRow, idxPixel):
	neighborScore = 0
	# print(idxPixelRow)
	# print(idxPixel)
	#Pixel ←
	if (idxPixel > 0 and valPixelRow[idxPixel - 1] > 100):
		neighborScore += 1
	#Pixel →
	if (idxPixel < 31 and valPixelRow[idxPixel + 1] > 100):
		neighborScore += 1
	#Pixel ↑
	if (idxPixelRow > 0 and valChar[idxPixelRow - 1][idxPixel] > 100):
		neighborScore += 1
	#Pixel ↓
	if (idxPixelRow < 31 and valChar[idxPixelRow + 1][idxPixel] > 100):
		neighborScore += 1
	#Pixel ↖
	if (idxPixelRow > 0 and idxPixel > 0 and valChar[idxPixelRow - 1][idxPixel - 1] > 100):
		neighborScore += 1
	#Pixel ↗
	if (idxPixelRow > 0 and idxPixel < 31 and valChar[idxPixelRow - 1][idxPixel + 1] > 100):
		neighborScore += 1
	#Pixel ↘
	if (idxPixelRow < 31 and idxPixel < 31 and valChar[idxPixelRow + 1][idxPixel + 1] > 100):
		neighborScore += 1
	#Pixel ↙
	if (idxPixelRow < 31 and idxPixel > 0 and valChar[idxPixelRow + 1][idxPixel - 1] > 100):
		neighborScore += 1
	return neighborScore

def pixelCalculator(degreeStart, idxPixelRow, idxPixel, direction):
	neighbor = np.array([])
	neighbor = np.append(neighbor, np.array([neighborScore(
		idxPixelRow - 1, idxPixel - 1)]))
	neighbor = np.append(neighbor, neighborScore(
		idxPixelRow - 1, idxPixel))
	neighbor = np.append(neighbor, neighborScore(
		idxPixelRow - 1, idxPixel + 1))
	neighbor = np.append(neighbor, neighborScore(
		idxPixelRow, idxPixel + 1))
	neighbor = np.append(neighbor, neighborScore(
		idxPixelRow + 1, idxPixel + 1))
	neighbor = np.append(neighbor, neighborScore(
		idxPixelRow + 1, idxPixel))
	neighbor = np.append(neighbor, neighborScore(
		idxPixelRow + 1, idxPixel - 1))
	neighbor = np.append(neighbor, neighborScore(
		idxPixelRow, idxPixel + 1))
	# if first pixel of shape
	if (degreeStart == 999):
		direction = np.random.choice(
			(np.where(neighbor == max(neighbor)))[0])
		degree = (direction + 1) * 45
		degreeStart = degree
	# if not the first pixel of shape
	else:
		if (direction == 0):
			allowedDirection = neighbor[7]
			indexAllowedDirection = [7]
			allowedDirection = np.append(
				allowedDirection, neighbor[0:2])
			indexAllowedDirection = np.append(
				indexAllowedDirection, [0,1])
		else:
			allowedDirection = neighbor[(
				direction - 1): (direction + 2)]
			indexAllowedDirection = [(direction - 1), (direction), (direction + 1)]
			if (direction == 7):
				allowedDirection = np.append(
					allowedDirection, neighbor[0])
				indexAllowedDirection[2] = 0
		direction = indexAllowedDirection[np.random.choice(
			(np.where(allowedDirection == max(allowedDirection)))[0])]
		degree = (direction + 1) * 45
		degreeStart = degree
		# Select the next pixel
	if (direction < 3):
		idxPixelRow = idxPixelRow - 1
	elif (direction > 3 and direction < 7):
		idxPixelRow = idxPixelRow - 1
	if(direction > 1 and direction < 5):
		idxPixel = idxPixel + 1
	elif(direction == 0 or direction > 5):
		idxPixel = idxPixel - 1
	if (neighborScore(idxPixelRow, idxPixel) > 1):
		return pixelCalculator(degreeStart, idxPixelRow, idxPixel, direction)
	else:
		return [idxPixelRow, idxPixel]
	# if (len(begin) > len(end)):
	# 	end.append

# load the input image from disk, convert it to grayscale, and blur
# it to reduce noise
org = cv2.imread("photo1.jpg")  # load image
gray = cv2.cvtColor(org, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)


# (thresh, blackWhiteImg) = cv2.threshold(grayImg, 127, 255, cv2.THRESH_BINARY)

"""
Show original, gray scale and black-and-white images:

cv2.imshow('Black white image', blackWhiteImg)
cv2.waitKey(0)
cv2.imshow('Black white image', grayImg)
cv2.waitKey(0)
cv2.imshow('Black white image', orgImg)
cv2.waitKey(0)
"""

# nz = cv2.findNonZero(blackWhiteImg)
# print(blackWhiteImg)

# perform edge detection, find contours in the edge map, and sort the
# resulting contours from left-to-right
edged = cv2.Canny(blurred, 30, 150)
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
						cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sort_contours(cnts, method="left-to-right")[0]

# initialize the list of contour bounding boxes and associated
# characters that we'll be OCR'ing
chars = []

# loop over the contours
for c in cnts:
	# compute the bounding box of the contour
	(x, y, w, h) = cv2.boundingRect(c)

	# filter out bounding boxes, ensuring they are neither too small
	# nor too large
	if (w >= 5 and w <= 150) and (h >= 15 and h <= 120):
		# extract the character and threshold it to make the character
		# appear as *white* (foreground) on a *black* background, then
		# grab the width and height of the thresholded image
		roi = gray[y:y + h, x:x + w]
		thresh = cv2.threshold(roi, 0, 255,
							   cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
		(tH, tW) = thresh.shape

		# if the width is greater than the height, resize along the
		# width dimension
		if tW > tH:
			thresh = imutils.resize(thresh, width=32)

		# otherwise, resize along the height
		else:
			thresh = imutils.resize(thresh, height=32)

		# re-grab the image dimensions (now that its been resized)
		# and then determine how much we need to pad the width and
		# height such that our image will be 32x32
		(tH, tW) = thresh.shape
		dX = int(max(0, 32 - tW) / 2.0)
		dY = int(max(0, 32 - tH) / 2.0)

		# pad the image and force 32x32 dimensions
		padded = cv2.copyMakeBorder(thresh, top=dY, bottom=dY,
									left=dX, right=dX, borderType=cv2.BORDER_CONSTANT,
									value=(0, 0, 0))
		padded = cv2.resize(padded, (32, 32))

		"""
		Extra array modification options

		# divide color with 255
		padded = padded.astype("float32") / 255.0
		# expand array
		padded = np.expand_dims(padded, axis=-1)
		"""

		# update our list of characters that will be OCR'd
		chars.append((padded, (x, y, w, h)))

# extract the bounding box locations and padded characters
boxes = [b[1] for b in chars]
chars = np.array([c[0] for c in chars], dtype="int32")

"""
Show image boxes

# loop over the predictions and bounding box locations together
for (x, y, w, h) in boxes:
	cv2.rectangle(org, (x, y), (x + w, y + h), (0, 255, 0), 2)

# show the image
cv2.imshow("Image", org)
cv2.waitKey(0)

"""

begin = []
end = []
# loop through found chars, which consist of 32x32 arrays
for idxChar, valChar in enumerate(chars):
	# loop through char, which consist of a array of 32
	for idxPixelRow, valPixelRow in enumerate(valChar):
		# loop through pixel row, which consist of 32 values
		for idxPixel, valPixel in enumerate(valPixelRow):
			if (valPixel > 100):
				scorePixel = neighborScore(idxPixelRow, idxPixel)
				# Register beginning pixel
				if (scorePixel > 3):
					begin.append([idxPixelRow, idxPixel])
					degreeStart = 999
					# give a neighbor score based on how many none white'ish pixels are arround it
					end.append(pixelCalculator(degreeStart, idxPixelRow, idxPixel, 0))
					print(begin)
					print(end)
					


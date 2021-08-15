import cv2 as cv
import numpy as np
import os

# Get the current project directory path
path = f'{os.path.dirname(os.path.abspath(__file__))}/'

img = cv.imread(f'{path}prac01ex04img01.png')

width, height = img.shape[:-1]

# Rotate
rotMat = cv.getRotationMatrix2D((width // 2, height // 2), 45, 1)
imgFixed = cv.warpAffine(img, rotMat, (width, height))

# Crop in
cropX = 95
cropY = 195
imgFixed = imgFixed[cropY:(width - cropY), cropX:(height - cropX)]

cv.imwrite(f'{path}prac01ex04img01fixed.png', imgFixed)

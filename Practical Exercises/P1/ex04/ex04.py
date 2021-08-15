import cv2 as cv
import numpy as np

relPath = 'Practical Exercises/P1/ex04/'

img = cv.imread(f'{relPath}prac01ex04img01.png')

width, height = img.shape[:-1]

# Rotate
rotMat = cv.getRotationMatrix2D((width // 2, height // 2), 45, 1)
imgFixed = cv.warpAffine(img, rotMat, (width, height))

# Crop in
cropX = 95
cropY = 195
imgFixed = imgFixed[cropY:(width - cropY), cropX:(height - cropX)]

cv.imwrite(f'{relPath}prac01ex04img01fixed.png', imgFixed)

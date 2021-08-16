import cv2 as cv
import numpy as np
import os

# Get the current project directory path
path = f'{os.path.dirname(os.path.abspath(__file__))}/'

img1Path = f'{path}prac02ex02img01.jpg'
img2Path = f'{path}prac02ex02img02.jpg'

img = cv.imread(f'{path}prac02ex02img01.jpg')

kernel = np.matrix([
    [0, 1, 0],
    [1, 1, 1],
    [0, 1, 0]
])
filtered = cv.filter2D(img, 1, kernel)

cv.imshow('Filtered', filtered)
cv.waitKey(0)
cv.destroyAllWindows()
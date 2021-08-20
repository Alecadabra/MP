import cv2 as cv
import numpy as np
import os

# Get the current project directory path
path = f'{os.path.dirname(os.path.abspath(__file__))}/'

def showImage(img, name):
    cv.imshow(name, img)
    cv.waitKey(0)
    cv.destroyAllWindows()

img = cv.imread(f'{path}prac02ex04img01.png')

showImage(img, 'Original')

newImg = np.zeros(img.shape, img.dtype)

# alpha in [1..3]
alpha = 1.3
# beta in [0..100]
beta = 40

height, width, channels = img.shape

for y in range(height):
    for x in range(width):
        for c in range(channels):
            newImg[y, x, c] = np.clip(alpha * img[y, x, c] + beta, 0, 255)

showImage(newImg, 'Fixed')

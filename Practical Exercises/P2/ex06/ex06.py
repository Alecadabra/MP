import cv2 as cv
import numpy as np
import os
from matplotlib import pyplot as plt
from numpy.__config__ import show

# Get the current project directory path
path = f'{os.path.dirname(os.path.abspath(__file__))}/'

def showImage(img, name):
    cv.imshow(name, img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def plotImage(img, name):
    imgFlat = img.flatten()
    hist, = np.histogram(imgFlat, 256, [0, 256])[:-1]
    cdf = hist.cumsum()
    cdfNormalised = cdf * hist.max() / cdf.max()

    plt.figure(name)
    plt.plot(cdfNormalised, color = 'b')
    plt.hist(imgFlat, 256, [0, 256], color = 'r')
    plt.xlim([0, 256])
    plt.legend(('cdf', 'histogram'), loc = 'upper left')
    plt.show()

img1 = cv.imread(f'{path}prac02ex06img01.png', cv.IMREAD_GRAYSCALE)
img2 = cv.imread(f'{path}prac02ex06img02.png', cv.IMREAD_GRAYSCALE)

size = (3, 3)
kernel = cv.getStructuringElement(cv.MORPH_RECT, size)

operations = [
    ('Original', lambda img: img),
    ('Erosion', lambda img: cv.erode(img, kernel)),
    ('Dilation', lambda img: cv.dilate(img, kernel)),
    ('Opening', lambda img: cv.dilate(cv.erode(img, kernel), kernel)),
    ('Closing', lambda img: cv.erode(cv.dilate(img, kernel), kernel)),
    ('Morphological gradient', lambda img: cv.dilate(img, kernel) - cv.erode(img, kernel)),
    ('Blackhat', lambda img: cv.erode(cv.dilate(img, kernel), kernel) - img)
]

for (name, op) in operations:
    for (i, img) in enumerate([img1, img2]):
        edited = op(img)
        showImage(edited, f'{name}: Image {i + 1}')
        
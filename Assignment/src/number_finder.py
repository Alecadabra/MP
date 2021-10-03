import cv2 as cv
import numpy as np
import os
from matplotlib import pyplot as plt

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

def task1():
    sep = os.path.sep
    # Get the current project directory path
    path = f'{os.path.dirname(os.path.abspath(__file__))}{sep}..{sep}'
    
    train = f'{path}train{sep}task1{sep}'

    trainImgs = [cv.imread(f'{train}BS{imgNum:02d}.jpg') for imgNum in range(1, 22)]

    for (i, img) in enumerate(trainImgs, start=1):
        #showImage(img, f'Image {i}')

        blurred = cv.GaussianBlur(img, (3, 3), 0)

        greyBlurred = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)

        edges = cv.Canny(greyBlurred, 300, 400)

        showImage(edges, f'Canny of img {i}')

        contours, hier = cv.findContours(edges, cv.CHAIN_APPROX_SIMPLE, )

    


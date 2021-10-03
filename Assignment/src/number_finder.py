import cv2 as cv
import numpy as np
import os
from matplotlib import pyplot as plt
import random

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

def randomColour():
    return (
        random.randint(0, 0xff),
        random.randint(0, 0xff),
        random.randint(0, 0xff)
    )

def aspectRatio(contour):
    x, y, width, height = cv.boundingRect(contour)

    if height == 0:
        # Divide by zero
        return float('inf')
    else:
        return width / height

def task1():
    sep = os.path.sep
    # Get the current project directory path
    path = f'{os.path.dirname(os.path.abspath(__file__))}{sep}..{sep}'
    
    train = f'{path}train{sep}task1{sep}'

    trainImgs = [cv.imread(f'{train}BS{imgNum:02d}.jpg') for imgNum in range(1, 22)]

    for (i, img) in enumerate(trainImgs, start=1):
        # showImage(img, f'Image {i}')

        blurred = cv.GaussianBlur(img, (3, 3), 0)

        greyBlurred = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)

        edges = cv.Canny(greyBlurred, 300, 400)

        # showImage(edges, f'Canny of img {i}')

        _, contours, _ = cv.findContours(edges, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

        drawnContours = img.copy()

        for i, _ in enumerate(contours):
            drawnContours = cv.drawContours(drawnContours, contours, i, randomColour(), thickness=2)

        # showImage(drawnContours, f'Contours for img {i}')

        hulls = [cv.convexHull(contour) for contour in contours]

        sortedHulls = sorted(hulls, key=lambda hull: aspectRatio(hull))[0:3]

        sortedContours = sorted(contours, key=lambda contour: abs(aspectRatio(contour) - 2))[0:3]

        drawnContours = img.copy()

        for i, _ in enumerate(sortedHulls):
            drawnContours = cv.drawContours(drawnContours, sortedContours, i, randomColour(), thickness=2)

        drawnRects = img.copy()

        for i, contour in enumerate(contours):
            x, y, w, h = cv.boundingRect(contour)
            drawnRects = cv.rectangle(drawnRects, (x, y), (x + w, y + h), color=randomColour(), thickness=2)
        
        showImage(drawnRects, f'Bounding rects for img {i}')



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

def solidity(contour):
    hull = cv.convexHull(contour)
    hullArea = cv.contourArea(hull)

    if hullArea == 0:
        # Divide by zero
        return float('inf')
    else:
        return cv.contourArea(contour) / hullArea

def task1():
    sep = os.path.sep
    # Get the current project directory path
    path = f'{os.path.dirname(os.path.abspath(__file__))}{sep}..{sep}'
    
    train = f'{path}train{sep}task2{sep}'

    trainImgs = [cv.imread(f'{train}DS{imgNum:02d}.jpg') for imgNum in range(1, 22)]

    for (n, img) in enumerate(trainImgs, start=1):
        # showImage(img, f'Image {i}')

        blurred = cv.GaussianBlur(img, (3, 3), 0)

        greyBlurred = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)

        edges = cv.Canny(greyBlurred, 300, 400)

        # showImage(edges, f'Canny of img {n}')

        # morphKernel = np.ones((7, 7), np.uint8)

        # morphed = cv.dilate(edges, morphKernel)

        # showImage(morphed, f'Morphed canny of img {n}')

        _, contours, _ = cv.findContours(edges, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

        imgWidth, imgHeight, _ = img.shape
        bigEnough = lambda contour: cv.contourArea(contour) > 0.0003 * (imgWidth * imgHeight)

        filteredContours = list(filter(bigEnough, contours))

        # sortedContours = sorted(filteredContours, key=lambda contour: abs(solidity(contour) - 2), reverse=True)

        # for i, contour in enumerate(sortedContours):
        #     drawnContour = img.copy()
        #     showImage(cv.drawContours(drawnContour, sortedContours, i, randomColour(), thickness=3), f'Img {n} contour {i + 1}')

        drawnContours = img.copy()

        for i, _ in enumerate(filteredContours):
            drawnContours = cv.drawContours(drawnContours, filteredContours, i, randomColour(), thickness=2)

        showImage(drawnContours, f'Contours for img {n}')

        hulls = [cv.convexHull(contour) for contour in contours]

        sortedHulls = sorted(hulls, key=lambda hull: aspectRatio(hull))[0:3]

        drawnHulls = img.copy()

        for i, _ in enumerate(hulls):
            drawnHulls = cv.drawContours(drawnHulls, hulls, i, randomColour(), thickness=2)

        # showImage(drawnHulls, f'Hulls for img {n}')

        drawnRects = img.copy()

        for i, contour in enumerate(contours):
            x, y, w, h = cv.boundingRect(contour)
            drawnRects = cv.rectangle(drawnRects, (x, y), (x + w, y + h), color=randomColour(), thickness=2)
        
        # showImage(drawnRects, f'Bounding rects for img {n}')





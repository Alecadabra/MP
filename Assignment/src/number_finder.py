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
    _, _, width, height = cv.boundingRect(contour)

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

def drawContours(img, contours, thickness=2):
    destImg = img.copy()

    for i in range(len(contours)):
        destImg = cv.drawContours(destImg, contours, i, randomColour(), thickness=thickness)

    return destImg

def findEdges(img):
    blurred = cv.GaussianBlur(img, (3, 3), 0)
    greyBlurred = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(greyBlurred, 300, 400)

    return edges

def relativeSize(img, contour):
    imgWidth, imgHeight, _ = img.shape
    _, _, width, height = cv.boundingRect(contour)

    return (width * height) / (imgWidth * imgHeight)

def findSimilar(rects):
    similarity = []

    sortedRects = list(sorted(rects, key=lambda r: r[3]))

    for i in range(len(sortedRects)):
        if i + 1 == len(rects):
            j = 0
        else:
            j = i + 1
        
        ci = rects[i]
        cj = rects[j]

        _, _, _, hi = ci
        _, _, _, hj = cj

        similarity.append(abs(hi - hj))
    
    return similarity
        

def task1():
    sep = os.path.sep
    # Get the current project directory path
    path = f'{os.path.dirname(os.path.abspath(__file__))}{sep}..{sep}'
    
    train = f'{path}train{sep}task1{sep}'

    trainImgs = [cv.imread(f'{train}BS{imgNum:02d}.jpg') for imgNum in range(1, 22)]

    for (n, img) in enumerate(trainImgs, start=1):

        # showImage(img, f'Image {n}')

        edges = findEdges(img)

        # showImage(edges, f'Canny of img {n}')

        _, contours, _ = cv.findContours(edges, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

        imgWidth, imgHeight, _ = img.shape

        filteredContours = list(filter(lambda c: relativeSize(img, c) > 0.0006, contours))

        filteredContours = list(filter(lambda c: 0.4 < aspectRatio(c) < 0.8, filteredContours))

        # filteredContours = list(sorted(filteredContours, key=lambda c: relativeSize(img, c), reverse=True))[:3]

        # sortedContours = sorted(filteredContours, key=lambda contour: abs(solidity(contour) - 2), reverse=True)

        drawnContours = drawContours(img, filteredContours)
        # showImage(drawnContours, f'Contours for img {n}')

        # hulls = [cv.convexHull(contour) for contour in contours]
        # drawnHulls = drawContours(img, hulls)
        # showImage(drawnHulls, f'Hulls for img {n}')

        rects = [cv.boundingRect(c) for c in filteredContours]

        similar = findSimilar(rects)

        sortedRects = list(sorted(
            rects,
            key=lambda r: similar[rects.index(r)]
        ))[:3]

        drawnRects = img.copy()
        for i, (x, y, w, h) in enumerate(sortedRects):
            drawnRects = cv.rectangle(drawnRects, (x, y), (x + w, y + h), color=randomColour(), thickness=2)
        showImage(drawnRects, f'Bounding rects for img {n}')

        ### (x, y), (width, height), angle = cv.minAreaRec
        # rotRects = [np.int0(cv.boxPoints(cv.minAreaRect(contour))) for contour in contours]
        # drawnRotRects = drawContours(img, rotRects)
        # showImage(drawnRotRects, f'Min area rects for img {n}')

        # approxCurves = [cv.approxPolyDP(contour, 0.05 * cv.arcLength(contour, True), True) for contour in contours]
        # drawnApproxCurves = drawContours(img, approxCurves)
        # showImage(drawnApproxCurves, f'Approx curves for img {n}')


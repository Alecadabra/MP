from types import LambdaType
from typing import Iterable
import cv2 as cv
import numpy as np
import os
from matplotlib import pyplot as plt
import random
from itertools import permutations

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

def rectangularity(contour):
    box = boxContour(contour)
    boxArea = cv.contourArea(box)

    if boxArea == 0:
        # Divide by zero
        return 0
    else:
        return cv.contourArea(contour) / boxArea

def boxContour(contour):
    x, y, w, h = cv.boundingRect(contour)
    return np.int0([(x, y), (x+w, y), (x+w, y+h), (x, y+h)])

def drawContours(img, contours, thickness=2):
    destImg = img.copy()

    for i in range(len(contours)):
        destImg = cv.drawContours(destImg, contours, i, randomColour(), thickness=thickness)

    return destImg

def findEdges(img, gaussian=3, t1=300, t2=400):
    blurred = cv.GaussianBlur(img, (gaussian, gaussian), 0)
    greyBlurred = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(greyBlurred, t1, t2)

    return edges

def relativeSize(img, contour):
    imgWidth, imgHeight = img.shape[:2]
    _, _, width, height = cv.boundingRect(contour)

    return (width * height) / (imgWidth * imgHeight)

# Find permutation with smallest differences in y values
# Finds the average differences between elements in an iterable, given 
# a mapping function
def avgDifference(iterable: Iterable, key: LambdaType):
    return sum(map(
        lambda x: key(x),
        permutations(iterable, 2)
    )) / len(list(permutations(iterable, 2)))

def cropImg(img, rotRect):
    _, (width, height), _ = rotRect
    rotBoundingCont = np.int0(cv.boxPoints(rotRect))
    points = rotBoundingCont.astype('float32')
    dest = np.array(
        [
            [0, height - 1],
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
        ],
        dtype='float32'
    )

    perspectiveMat = cv.getPerspectiveTransform(points, dest)
    cropped = cv.warpPerspective(img, perspectiveMat, (width, height))

    return cropped

def findNumbers(edges, relSizeThresh=0.0006, minRatio=0.4, maxRatio=0.8):
    _, contours, _ = cv.findContours(edges, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    filteredContours = list(filter(lambda c: relativeSize(edges, c) > relSizeThresh, contours))

    filteredContours = list(filter(lambda c: minRatio < aspectRatio(c) < maxRatio, filteredContours))

    rects = [cv.boundingRect(c) for c in filteredContours]

    sortedRects = rects
    sortedRects = sorted(sortedRects, key=lambda c: c[3])

    perms = permutations(rects, 3)
    perms = list(filter(
        lambda p: p[0][0] < p[1][0] < p[2][0] and (p[0][0] + p[0][2]) < (p[1][0] + p[1][2]) < (p[2][0] + p[2][2]),
        perms
    ))
    perms = list(filter(
        lambda p: 
            avgDifference(p, lambda pp: abs(pp[0][1] - pp[1][1])) < 10 and
            avgDifference(p, lambda pp: abs((pp[0][1] + pp[0][3]) - (pp[1][1] + pp[1][3]))) < 10,
        perms
    ))

    perm = max(
        perms,
        key=lambda p: (p[0][2] * p[0][3]) + (p[1][2] * p[1][3]) + (p[2][2] * p[2][3])
    )

    return perm

def numberAngle(numbers):
    angleOpp = ((numbers[2][0] + numbers[2][2]) - numbers[0][0])
    angleAdj = ((numbers[2][1] + numbers[2][3]) - (numbers[0][1] + numbers[0][3]))

    if angleAdj != 0:
        angle = np.arctan(angleOpp / angleAdj)
    else:
        angle = 0
    
    return angle

def cropToNumbers(img):
    edges = findEdges(img)

    numberRects = findNumbers(edges)

    # drawnRects = img.copy()
    # for i, (x, y, w, h) in enumerate(numberRects):
    #     drawnRects = cv.rectangle(drawnRects, (x, y), (x + w, y + h), color=randomColour(), thickness=2)
    # showImage(drawnRects, f'Bounding rects for img {n}')

    angle = numberAngle(numberRects)

    pad = int((sum(map(lambda p: p[3], numberRects)) / 3) * 0.25)
    boundingRect = (
        numberRects[0][0] - pad,
        numberRects[0][1] - pad,
        (numberRects[2][0] + numberRects[2][2]) - numberRects[0][0] + pad + pad,
        max(numberRects, key=lambda p: p[3])[3] + pad + pad
    )

    bx, by, bw, bh = boundingRect
    # drawnBounding = img.copy()
    # drawnBounding = cv.rectangle(drawnBounding, (bx, by), (bx+bw, by+bh), randomColour(), thickness=2)
    # showImage(drawnBounding, f'Enclosing box for img {n}, angle: {angle}')

    rotBounding = (bx+(bw/2), by+(bh/2)), (bw, bh), angle
    # rotBoundingCont = np.int0(cv.boxPoints(((bx+(bw/2), by+(bh/2)), (bw, bh), angle)))
    # drawnRotBounding = drawContours(img, [rotBoundingCont])
    # showImage(drawnRotBounding, f'Rotated enclosing box for img {n}')

    # rotatedImg = cropImg(img.copy(), angle)
    # showImage(rotatedImg, f'Rotated enclosing box for img {n}')

    cropped = cropImg(img, rotBounding)

    return cropped

def task1(): 
    sep = os.path.sep
    # Get the current project directory path
    path = f'{os.path.dirname(os.path.abspath(__file__))}{sep}..{sep}'
    
    train = f'{path}train{sep}task1{sep}'

    output = f'{path}output{sep}task1{sep}'

    trainImgs = [cv.imread(f'{train}BS{imgNum:02d}.jpg') for imgNum in range(1, 22)]

    for (n, img) in enumerate(trainImgs, start=1):

        cropped = cropToNumbers(img)

        # showImage(cropped, f'Cropped & rotated img {n}')
        cv.imwrite(f'{output}DetectedArea{n:02d}.jpg', cropped)



# showImage(img, f'Image {n}')

# boxEdges = findEdges(img, t1=30, t2=50)
# showImage(boxEdges, f'Edges for img {n}')

# _, boxContours, _ = cv.findContours(boxEdges, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
# filteredBoxes = boxContours
# # filteredBoxes = list(filter(lambda c: relativeSize(img, c) > 0.002, filteredBoxes))
# filteredBoxes = list(filter(lambda c: rectangularity(c) > 0.7, filteredBoxes))
# # filteredBoxes = list(filter(lambda c: 1.2 < aspectRatio(c) < 1.8, filteredBoxes))

# drawnBox = drawContours(img, filteredBoxes)
# showImage(drawnBox, f'Box for img {n}')


# filteredContours = list(sorted(filteredContours, key=lambda c: relativeSize(img, c), reverse=True))[:3]

# sortedContours = sorted(filteredContours, key=lambda contour: abs(solidity(contour) - 2), reverse=True)

# drawnContours = drawContours(img, filteredContours)
# showImage(drawnContours, f'Contours for img {n}')

# hulls = [cv.convexHull(contour) for contour in contours]
# drawnHulls = drawContours(img, hulls)
# showImage(drawnHulls, f'Hulls for img {n}')


### (x, y), (width, height), angle = cv.minAreaRec
# rotRects = [np.int0(cv.boxPoints(cv.minAreaRect(contour))) for contour in contours]
# drawnRotRects = drawContours(img, rotRects)
# showImage(drawnRotRects, f'Min area rects for img {n}')

# approxCurves = [cv.approxPolyDP(contour, 0.05 * cv.arcLength(contour, True), True) for contour in contours]
# drawnApproxCurves = drawContours(img, approxCurves)
# showImage(drawnApproxCurves, f'Approx curves for img {n}')
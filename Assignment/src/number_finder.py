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
    return tuple([random.randint(0, 0xff) for _ in range(3)])

def aspectRatio(contour):
    _, _, width, height = cv.boundingRect(contour)

    return width / height if height != 0 else 0

def solidity(contour):
    hull = cv.convexHull(contour)
    hullArea = cv.contourArea(hull)

    return cv.contourArea(contour) / hullArea if hullArea != 0 else 0

def rectangularity(contour):
    box = boxContour(contour)
    boxArea = cv.contourArea(box)
    
    return cv.contourArea(contour) / boxArea if boxArea != 0 else 0

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
    _, _, cntWidth, cntHeight = cv.boundingRect(contour)

    imgArea = imgWidth * imgHeight
    contourArea = cntWidth * cntHeight

    return contourArea / imgArea if imgArea != 0 else 0

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

    filteredContours = [c for c in contours if relativeSize(edges, c) > relSizeThresh]

    filteredContours = [c for c in filteredContours if minRatio < aspectRatio(c) < maxRatio]

    rects = [cv.boundingRect(c) for c in filteredContours]

    sortedRects = rects
    sortedRects = sorted(sortedRects, key=lambda c: c[3])

    perms = permutations(rects, 3)
    # Filter to only get left to right perms
    perms = [p for p in perms if p[0][0] < p[1][0] < p[2][0] and
        (p[0][0] + p[0][2]) < (p[1][0] + p[1][2]) < (p[2][0] + p[2][2])
    ]

    # Filter to only get perms of similar heights and y values to each other
    # Loop until the filtered list is non-empty
    filteredPerms = []
    thresh = 10
    while (len(filteredPerms) == 0):
        filteredPerms = [p for p in perms if 
            avgDifference(p, lambda pp: abs(pp[0][1] - pp[1][1])) < thresh and
            avgDifference(p, lambda pp: abs((pp[0][1] + pp[0][3]) - (pp[1][1] + pp[1][3]))) < thresh
        ]
        thresh += 1

    perm = max(
        filteredPerms,
        key=lambda p: (p[0][2] * p[0][3]) + (p[1][2] * p[1][3]) + (p[2][2] * p[2][3])
    )

    return perm

def numberAngle(numbers):
    angleOpp = ((numbers[2][0] + numbers[2][2]) - numbers[0][0])
    angleAdj = ((numbers[2][1] + numbers[2][3]) - (numbers[0][1] + numbers[0][3]))
    
    return np.arctan(angleOpp / angleAdj) if angleAdj != 0 else 0

def padRect(rect, padX, padY):
    padX = int(padX)
    padY = int(padY)
    boundingRect = (
        rect[0] - padX,
        rect[1] - padY,
        rect[2] + padX * 2,
        rect[3] + padY * 2
    )
    return boundingRect

def cropToNumbers(img):
    edges = findEdges(img)

    numberRects = findNumbers(edges)

    # drawnRects = img.copy()
    # for i, (x, y, w, h) in enumerate(numberRects):
    #     drawnRects = cv.rectangle(drawnRects, (x, y), (x + w, y + h), color=randomColour(), thickness=2)
    # showImage(drawnRects, f'Bounding rects for img {n}')

    angle = numberAngle(numberRects)

    pad = int((sum([p[3] for p in numberRects]) / 3) * 0.25)
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
    projDir = f'{os.path.dirname(os.path.abspath(__file__))}{sep}..{sep}'
    
    trainDir = f'{projDir}train{sep}task1{sep}'

    outputDir = f'{projDir}output{sep}task1{sep}'

    trainImgs = [cv.imread(f'{trainDir}BS{imgNum:02d}.jpg') for imgNum in range(1, 22)]

    digitsDir = f'{projDir}digits{sep}'

    digitsName = {
        0: 'Zero',
        1: 'One',
        2: 'Two',
        3: 'Three',
        4: 'Four',
        5: 'Five',
        6: 'Six',
        7: 'Seven',
        8: 'Eight',
        9: 'Nine',
        'l': 'LeftArrow',
        'r': 'RightArrow'
    }

    digits = {
        key: [cv.imread(f'{digitsDir}{name}{i}.jpg') for i in range(1,6)] for key, name in digitsName.items()
    }

    for (n, img) in enumerate(trainImgs, start=1):

        cropped = cropToNumbers(img)

        # showImage(cropped, f'Cropped & rotated img {n}')
        cv.imwrite(f'{outputDir}DetectedArea{n:02d}.jpg', cropped)

        numberRects = findNumbers(findEdges(cropped), relSizeThresh=0.006)

        # drawnRects = cropped.copy()
        # for i, (x, y, w, h) in enumerate(numbers):
        #     drawnRects = cv.rectangle(drawnRects, (x, y), (x + w, y + h), color=randomColour(), thickness=2)
        # showImage(drawnRects, f'Bounding rects for img {n}')

        minWidth, minHeight = 28, 40

        numberRectsPadded = [padRect(rect, (rect[3] * 0.05), (rect[3] * 0.05)) for rect in numberRects]

        def rectToRotRect(rect):
            x, y, w, h = rect
            return (x+(w/2), y+(h/2)), (w, h), 0

        numberImgs = [cropImg(cropped, rectToRotRect(numImg)) for numImg in numberRectsPadded]

        resizedNumberImgs = [cv.resize(numImg, (28, 40)) for numImg in numberImgs]

        for i, numberImg in enumerate(resizedNumberImgs, start=1):
            showImage(numberImg, f'Img {n} number {i}')

    # for digit, l in digits.items():
    #     for i, digitImg in enumerate(l):
    #         showImage(digitImg, f'Digit {digit} img {i + 1}')




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
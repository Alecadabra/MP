from types import LambdaType
from typing import Any, Dict, Iterable, List, Tuple, Union
import cv2 as cv
import numpy as np
import os
from matplotlib import pyplot as plt
import random
from itertools import permutations
from statistics import median, mean
from collections import OrderedDict

def showImage(img, name: str):
    '''Shows an image mat using cv.imshow.'''
    cv.imshow(name, img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def randomColour() -> Tuple[int, int, int]:
    '''Randomly generates a colour as a tuple of three values from 0 to 255.'''
    return tuple([random.randint(0x80, 0xff) for _ in range(3)])

def aspectRatio(contour):
    '''Computes the aspect ratio of a contour's bounding rectange
    (width / height).'''
    _, _, width, height = cv.boundingRect(contour)

    return width / height if height != 0 else 0

def findEdges(img, gaussian=3, t1=300, t2=400):
    '''Applys the Canny edge detector to an image with preset values and
    preprocessing.'''
    blurred = cv.GaussianBlur(img, (gaussian, gaussian), 0)
    greyBlurred = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(greyBlurred, t1, t2)

    return edges

def relativeSize(img, contour):
    '''Computes the relative size of a contour as a proportion of the source
    image size.'''
    imgWidth, imgHeight = img.shape[:2]
    _, _, cntWidth, cntHeight = cv.boundingRect(contour)

    imgArea = imgWidth * imgHeight
    contourArea = cntWidth * cntHeight

    return contourArea / imgArea if imgArea != 0 else 0

def avgDifference(iterable: Iterable, key: LambdaType):
    '''Finds the average differences between elements in an iterable, given 
    a mapping function'''
    total = sum([key(x[0], x[1]) for x in permutations(iterable, 2)])
    numElements = len(list(permutations(iterable, 2)))

    return total / numElements

def cropImg(img, rotRect):
    '''Crops a given image mat to a rotated rectange, using it's angle.
    `rotRect` can be sourced from cv.minAreaRec and is a tuple of format
    `(x, y), (width, height), angle`.'''
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
    '''The full algorithm to find the three building numbers from an image,
    as a tuple of bounding rectangles (Tuples of `x, y, width, height`).'''
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
            avgDifference(p, lambda a, b: abs(a[1] - b[1])) < thresh and
            avgDifference(p, lambda a, b: abs((a[1] + a[3]) - (b[1] + b[3]))) < thresh
        ]
        thresh += 1

    perm = max(
        filteredPerms,
        key=lambda p: (p[0][2] * p[0][3]) + (p[1][2] * p[1][3]) + (p[2][2] * p[2][3])
    )

    return perm

def numberAngle(numbers: Tuple):
    '''Takes in a tuple of three bounding rects and finds the angle they slant
    at.'''
    angleOpp = ((numbers[2][0] + numbers[2][2]) - numbers[0][0])
    angleAdj = ((numbers[2][1] + numbers[2][3]) - (numbers[0][1] + numbers[0][3]))
    
    return np.arctan(angleOpp / angleAdj) if angleAdj != 0 else 0

def padRect(rect, padX, padY):
    '''Takes in a bounding rectangle and pads it on x and y.'''
    padX = round(padX)
    padY = round(padY)
    boundingRect = (
        rect[0] - padX,
        rect[1] - padY,
        rect[2] + padX * 2,
        rect[3] + padY * 2
    )
    return boundingRect

def cropToNumbers(img):
    '''The full algorithm to take in an edges image and crop it to just the
    area of the numbers. To actually find the numbers it uses `findNumbers`.'''
    edges = findEdges(img)

    numberRects = findNumbers(edges)

    # drawnRects = img.copy()
    # for i, (x, y, w, h) in enumerate(numberRects):
    #     drawnRects = cv.rectangle(drawnRects, (x, y), (x + w, y + h), color=randomColour(), thickness=2)
    # showImage(drawnRects, f'Bounding rects for img {n}')

    angle = numberAngle(numberRects)

    pad = round((sum([p[3] for p in numberRects]) / 3) * 0.25)
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

def matchNum(img, digits):
    '''Takes in a source image and map from desired values to lists of image
    templates, and uses it to classify the source image as one of the digits.'''
    method = cv.TM_CCOEFF_NORMED
    maxima = {value: 0 for value in digits.keys()}

    img = cv.cvtColor(img.copy(), cv.COLOR_BGR2GRAY)

    for key, templates in digits.items():
        maxMatch = 0

        for template in templates:
            template = cv.cvtColor(template, cv.COLOR_BGR2GRAY)

            matchImg = img.copy()
            matchRes = cv.matchTemplate(matchImg, template, method)
            _, resValue, _, _ = cv.minMaxLoc(matchRes)

            maxMatch = max(maxMatch, resValue)
        
        maxima[key] = maxMatch
    
    maxKey, _ = max(maxima.items(), key=lambda item: item[1])

    return maxKey
        
def rectToRotRect(rect):
    '''Converts a rect to a rotated rect'''
    x, y, w, h = rect
    return (x+(w/2), y+(h/2)), (w, h), 0

def classifyRects(cropped, numberRects, digits):
        # drawnRects = cropped.copy()
        # for i, (x, y, w, h) in enumerate(numbers):
        #     drawnRects = cv.rectangle(drawnRects, (x, y), (x + w, y + h), color=randomColour(), thickness=2)
        # showImage(drawnRects, f'Bounding rects for img {n}')

        numberRectsPadded = [padRect(rect, (rect[3] * 0.08), (rect[3] * 0.08)) for rect in numberRects]

        # Defined by size of images in digits directory
        minWidth, minHeight = 28, 40
        digitsRatio = minWidth / minHeight

        # Pad for aspect ratio
        for i, numberRect in enumerate(numberRectsPadded):
            _, _, w, h = numberRect
            ratio = w / h

            if ratio != digitsRatio:
                if ratio < digitsRatio:
                    padX = (h * digitsRatio) - w
                    numberRectsPadded[i] = padRect(numberRect, padX, 0)
                else:
                    padY = (w / digitsRatio) - h
                    numberRectsPadded[i] = padRect(numberRect, 0, padY)


        # Use the rects to crop out the number images
        numberImgs = [cropImg(cropped, rectToRotRect(numImg)) for numImg in numberRectsPadded]

        # Scale to fit the matcher digits images
        resizedNumberImgs = [cv.resize(numImg, (minWidth, minHeight)) for numImg in numberImgs]

        actualNumbers = [matchNum(numberImg, digits) for numberImg in resizedNumberImgs]

        return tuple(actualNumbers)

def classifyRectsDirectional(cropped, rectGroups: List[List[tuple]], digits: Dict[int, Any], arrows: Dict[str, Any]) -> Tuple[List[List[int]], List[str]]:
    numberRectGroupsPadded = [[padRect(pp, pp[3] * 0.08, pp[3] * 0.08) for pp in p] for p in rectGroups]

    # Add the arrow box
    def arrowBox(p: tuple) -> tuple:
        a, _, c = p
        floatRect = (
            c[0] + c[2] + (mean(pp[2] for pp in p)) * 0.3,
            mean(pp[1] for pp in p),
            (c[0] - a[0]) * 0.6,
            mean(pp[2] for pp in p) * 1.4
        )
        return tuple(round(pp) for pp in floatRect)
    numberRectGroupsPadded = [(p[0], p[1], p[2], arrowBox(p)) for p in numberRectGroupsPadded]
    
    # Defined by size of images in digits directory
    minWidth, minHeight = 28, 40
    digitsRatio = minWidth / minHeight

    actualNumbersGroup = []
    directionsGroup = []
    visited = []

    for numberRectsPadded in numberRectGroupsPadded:
        newRects = []

        if numberRectsPadded[0][:2] not in visited:

            visited.append(numberRectsPadded[0][:2])

            for numberRect in numberRectsPadded:
                _, _, w, h = numberRect
                ratio = w / h

                if ratio != digitsRatio:
                    if ratio < digitsRatio:
                        padX = (h * digitsRatio) - w
                        newRects.append(padRect(numberRect, padX, 0))
                    else:
                        padY = (w / digitsRatio) - h
                        newRects.append(padRect(numberRect, 0, padY))
                else:
                    newRects.append(numberRect)
                
            # Use the rects to crop out the number images
            numberImgs = [cropImg(cropped, rectToRotRect(rect)) for rect in newRects]

            # Scale to fit the matcher digits images
            resizedNumberImgs = [cv.resize(numImg, (minWidth, minHeight)) for numImg in numberImgs]

            # Classify numbers
            actualNumbers = [matchNum(numberImg, digits) for numberImg in resizedNumberImgs[:-1]]

            # Evil hack - replace middle 6's with 0's
            actualNumbers = actualNumbers if actualNumbers[1] != 6 else (actualNumbers[0], 0, actualNumbers[2])
            
            direction = matchNum(resizedNumberImgs[-1], arrows)
            directionStr = 'left' if direction == 'l' else 'right'

            if actualNumbers not in actualNumbersGroup:
                actualNumbersGroup.append(actualNumbers)
                directionsGroup.append(directionStr)
    
    return actualNumbersGroup, directionsGroup

def task1(testImgs: List, outputDir: str, digitsDict: Dict): 
    for n, img in enumerate(testImgs, start=1):
        cropped = cropToNumbers(img)

        # showImage(cropped, f'Cropped & rotated img {n}')
        cv.imwrite(f'{outputDir}DetectedArea{n:02d}.jpg', cropped)

        numberRects = findNumbers(findEdges(cropped), relSizeThresh=0.006)

        digits = {k: v for k, v in digitsDict.items() if k != 'l' and k != 'r'}

        actualNumbers = classifyRects(cropped, numberRects, digits)

        with open(f'{outputDir}Building{n:02d}.txt', 'w') as file:
            a, b, c = actualNumbers
            file.write(f'Building {a}{b}{c}')
            # showImage(img, f'{a}{b}{c}')

def findNumbersDirectional(edges, relSizeThresh=0.0003, minRatio=0.4, maxRatio=0.8):
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
    imgThresh = 0
    while len(filteredPerms) == 0:
        imgThresh += 1

        filteredPerms = [p for p in perms if 
            avgDifference(p, lambda a, b: abs(a[1] - b[1])) < imgThresh and
            avgDifference(p, lambda a, b: abs((a[1] + a[3]) - (b[1] + b[3]))) < imgThresh and
            p[0][0] + p[0][2] - p[1][0] < imgThresh and
            p[1][0] + p[1][2] - p[2][0] < imgThresh
        ]

    # Sort by y value
    filteredPerms = sorted(filteredPerms, key=lambda p: p[0][1])

    return filteredPerms

def cropToNumbersDirectional(img):
    edges = findEdges(img)

    numberRectGroups = findNumbersDirectional(edges)

    pad = round((sum(sum(pp[3] for pp in p) / 3 for p in numberRectGroups) / len(numberRectGroups)) * 0.1)
    firstRects = numberRectGroups[0]
    lastRects = numberRectGroups[len(numberRectGroups) - 1]
    boundingRect = (
        round(median(p[0][0] for p in numberRectGroups)) - pad,
        round(mean(pp[1] for pp in firstRects) - pad),
        round((lastRects[2][0] + lastRects[2][2]) - firstRects[0][0] + pad * 2 + firstRects[0][2] * 2.5),
        lastRects[2][1] + lastRects[2][3] - firstRects[0][1] + pad * 2
    )

    bx, by, bw, bh = boundingRect

    angle = mean(numberAngle(p) for p in numberRectGroups)

    rotBounding = (bx+(bw/2), by+(bh/2)), (bw, bh), angle

    cropped = cropImg(img.copy(), rotBounding)

    return cropped

def task2(testImgs: List, outputDir: str, digitsDict: Dict):
    for n, img in enumerate(testImgs, start=1):

        cropped = cropToNumbersDirectional(img)
        
        cv.imwrite(f'{outputDir}DetectedArea{n:02d}.jpg', cropped)

        numberRectGroups = findNumbersDirectional(findEdges(cropped))
        
        digits = {k: v for k, v in digitsDict.items() if k != 'l' and k != 'r'}
        arrows = {k: v for k, v in digitsDict.items() if k == 'l' or k == 'r'}

        actualNumbersGroup, directionsGroup = classifyRectsDirectional(cropped, numberRectGroups, digits, arrows)

        with open(f'{outputDir}Building{n:02d}.txt', 'w') as file:
            for actualNumbers, direction in zip(actualNumbersGroup, directionsGroup):
                a, b, c = actualNumbers
                file.write(f'Building {a}{b}{c} to the {direction}\n')

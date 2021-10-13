from types import LambdaType
from typing import Any, Dict, Iterable, List, Tuple, Union
import cv2 as cv
import numpy as np
import random
from itertools import permutations
from statistics import median, mean

# Tasks ------------------------------------------------------------------------

def task1(
    testImgs: List[Any],
    outputDir: str,
    templatesDict: Dict[Union[int, str], Any]
) -> None:
    '''The structure for task1 - take in a list of images, and find the
    area containing the building number, and classify the building number
    itself'''

    for n, img in enumerate(testImgs, start=1):

        # Crop the image to the area containing the building number
        cropped = _cropToNumbers(img)

        cv.imwrite(f'{outputDir}DetectedArea{n:02d}.jpg', cropped)

        # Find the numbers within the cropped area
        numberRects = _findNumbers(_findEdges(cropped), relSizeThresh=0.006)

        # Filter the templates to only include digits
        digits = {
            k: v for k, v in templatesDict.items() if k != 'l' and k != 'r'
        }

        # Classify the numbers
        actualNumbers = _classifyRects(cropped, numberRects, digits)

        with open(f'{outputDir}Building{n:02d}.txt', 'w') as file:
            a, b, c = actualNumbers
            file.write(f'Building {a}{b}{c}')

def task2(
    testImgs: List[Any],
    outputDir: str,
    templatesDict: Dict[Union[int, str], Any]
) -> None:
    '''The structure for task2 - take in images of multiple building numbers
    with arrows, find the area containing these signs, and the values and
    directions of the numbers and arrows respectively.'''

    for n, img in enumerate(testImgs, start=1):

        # Crop the image to the sign area
        cropped = _cropToNumbersDirectional(img)
        
        cv.imwrite(f'{outputDir}DetectedArea{n:02d}.jpg', cropped)

        # Find the groups of bounding boxes around the digits and arrows
        numberRectGroups = _findNumbersDirectional(_findEdges(cropped))
        
        # Split the digitsDict into seperate dicts for digits and arrows
        digits = {
            k: v for k, v in templatesDict.items() if k != 'l' and k != 'r'
        }
        arrows = {
            k: v for k, v in templatesDict.items() if k == 'l' or k == 'r'
        }

        # Classify the numbers and directions
        actualNumbersGroups, directions = _classifyRectsDirectional(
            cropped, numberRectGroups, digits, arrows
        )

        with open(f'{outputDir}Building{n:02d}.txt', 'w') as file:
            for actualNumbers, direction in zip(
                actualNumbersGroups, directions
            ):
                a, b, c = actualNumbers
                file.write(f'Building {a}{b}{c} to the {direction}\n')

# Find number boxes ------------------------------------------------------------

def _findNumbers(
    edges: Any,
    relSizeThresh=0.0006,
    minRatio=0.4,
    maxRatio=0.8
) -> Tuple[tuple, tuple, tuple]:
    '''The full algorithm to find the three building numbers from an image,
    as a tuple of bounding rectangles (Tuples of `x, y, width, height`).'''

    _, contours, _ = cv.findContours(
        edges, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE
    )

    # Reduce the size by filtering out small contours, and ones far from the
    # desired aspect ratio
    contours = [c for c in contours if _relativeSize(edges, c) > relSizeThresh]
    contours = [c for c in contours if minRatio < _aspectRatio(c) < maxRatio]

    # Map contours to their bounding rectangles
    rects = [cv.boundingRect(c) for c in contours]

    # Get all permutations of size 3 of the bounding boxes
    perms = permutations(rects, 3)
    # Filter to only get left to right perms
    perms = [p for p in perms if p[0][0] < p[1][0] < p[2][0] and
        (p[0][0] + p[0][2]) < (p[1][0] + p[1][2]) < (p[2][0] + p[2][2])
    ]

    # Filter to only get perms of similar heights and y values to each other
    # Loop until the filtered list is non-empty
    filteredPerms = []
    thresh = 6
    while (len(filteredPerms) == 0):
        filteredPerms = [p for p in perms if 
            _avgDiff(p, lambda a, b: abs(a[1] - b[1])) < thresh and
            _avgDiff(
                p, lambda a, b: abs((a[1] + a[3]) - (b[1] + b[3]))
            ) < thresh
        ]
        thresh += 1

    # After all of this filtering, we can assume the largest remaining
    # permutation is that of the building number
    perm = max(
        filteredPerms,
        key=lambda p: (
            (p[0][2] * p[0][3]) + (p[1][2] * p[1][3]) + (p[2][2] * p[2][3])
        )
    )

    return perm

def _findNumbersDirectional(
    edges: Any,
    relSizeThresh=0.0003,
    minRatio=0.4,
    maxRatio=0.8
) -> List[Tuple[Tuple[int, int, int, int]]]:
    '''The full algorithm to take in an edges image and find the bounding
    boxes around each of the separate building numbers, for use on a
    directional sign.'''

    _, contours, _ = cv.findContours(
        edges, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE
    )

    # Reduce the size by filtering out small contours, and ones far from the
    # desired aspect ratio
    contours = [c for c in contours if _relativeSize(edges, c) > relSizeThresh]
    contours = [c for c in contours if minRatio < _aspectRatio(c) < maxRatio]

    # Map contours to their bounding boxes
    rects = [cv.boundingRect(c) for c in contours]

    # Get all permutations of size 3 of the bounding boxes
    perms = permutations(rects, 3)

    # Filter to only get left to right perms
    perms = [p for p in perms if p[0][0] < p[1][0] < p[2][0] and
        (p[0][0] + p[0][2]) < (p[1][0] + p[1][2]) < (p[2][0] + p[2][2])
    ]

    # Filter to only get perms of similar heights and y values to each other,
    # and close to each other on x.
    # Loop until the filtered list is non-empty, increasing the threshold.
    filteredPerms = []
    thresh = 0
    while len(filteredPerms) == 0:
        thresh += 1

        filteredPerms = [p for p in perms if 
            _avgDiff(p, lambda a, b: abs(a[1] - b[1])) < thresh and
            _avgDiff(
                p, lambda a, b: abs((a[1] + a[3]) - (b[1] + b[3]))
            ) < thresh and
            p[0][0] + p[0][2] - p[1][0] < thresh and
            p[1][0] + p[1][2] - p[2][0] < thresh
        ]

    # Sort by y value
    filteredPerms = sorted(filteredPerms, key=lambda p: p[0][1])

    return filteredPerms

# Crop to number area ----------------------------------------------------------

def _cropToNumbers(img: Any) -> Any:
    '''The full algorithm to take in an image and crop it to just the
    area of the numbers. To actually find the numbers it uses `findNumbers`.'''

    edges = _findEdges(img)

    # Find the three bounding boxes surrounding the three numbers, in left
    # to right order
    numberRects = _findNumbers(edges)

    # Find the angle that these numbers are skewed at
    angle = _numberAngle(numberRects)

    # Pad the rectangle a bit
    pad = round((sum([p[3] for p in numberRects]) / 3) * 0.25)
    boundingRect = (
        numberRects[0][0] - pad,
        numberRects[0][1] - pad,
        (numberRects[2][0] + numberRects[2][2]) - numberRects[0][0] + pad + pad,
        max(numberRects, key=lambda p: p[3])[3] + pad + pad
    )

    # Rotate the rectangle by the computed angle
    bx, by, bw, bh = boundingRect
    rotBounding = (bx+(bw/2), by+(bh/2)), (bw, bh), angle

    # Crop the original image to this rotated rectangle
    cropped = _cropImg(img, rotBounding)

    return cropped

def _cropToNumbersDirectional(img: Any) -> Any:
    '''Full algorithm to take in an image of multiple directional building
    numbers and give the cropped area containing just the numbers and arrows.'''

    # Apply edge detection
    edges = _findEdges(img)

    # Find the groups of bounding boxes around the numbers
    numberRectGroups = _findNumbersDirectional(edges)

    # Create a bounding box around the entire group
    pad = round((sum(
        sum(pp[3] for pp in p) / 3 for p in numberRectGroups
    ) / len(numberRectGroups)) * 0.1)
    firstRects = numberRectGroups[0]
    lastRects = numberRectGroups[len(numberRectGroups) - 1]
    boundingRect = (
        round(median(p[0][0] for p in numberRectGroups)) - pad,
        round(mean(pp[1] for pp in firstRects) - pad),
        round((
            lastRects[2][0] + lastRects[2][2]
        ) - firstRects[0][0] + pad * 2 + firstRects[0][2] * 2.5),
        lastRects[2][1] + lastRects[2][3] - firstRects[0][1] + pad * 2
    )

    # Find the skew that the numbers are at
    angle = mean(_numberAngle(p) for p in numberRectGroups)

    # Define a rotated bounding box using this angle
    bx, by, bw, bh = boundingRect
    rotBounding = (bx+(bw/2), by+(bh/2)), (bw, bh), angle

    # Crop the image to this rotated bounding box
    cropped = _cropImg(img.copy(), rotBounding)

    return cropped

# Classification ---------------------------------------------------------------

def _classifyRects(
    cropped: Any,
    numberRects: List[Tuple[int, int, int, int]],
    digits: Dict[int, Any]
) -> Tuple[int]:
    '''The full algorithm to take in an area of a building number, the bounding
    boxes of the numbers, and a digit classificaiton map, and return the
    true digits.'''

    # Pad the number rectangles a bit
    numberRectsPadded = [(
        _padRect(rect, (rect[3] * 0.08), (rect[3] * 0.08))
    ) for rect in numberRects]

    # Defined by size of images in digits directory
    minW, minH = 28, 40
    templateRatio = minW / minH

    # Pad for aspect ratio
    for i, numberRect in enumerate(numberRectsPadded):
        _, _, w, h = numberRect
        ratio = w / h

        if ratio != templateRatio:
            if ratio < templateRatio:
                padX = (h * templateRatio) - w
                numberRectsPadded[i] = _padRect(numberRect, padX, 0)
            else:
                padY = (w / templateRatio) - h
                numberRectsPadded[i] = _padRect(numberRect, 0, padY)

    # Use the rects to crop out the number images
    numberImgs = [
        _cropImg(
            cropped, _rectToRotRect(numImg)
        ) for numImg in numberRectsPadded
    ]

    # Scale to fit the matcher digits images
    resizedNumberImgs = [
        cv.resize(numImg, (minW, minH)) for numImg in numberImgs
    ]

    # Run through the classifier
    actualNumbers = [
        _matchNum(numberImg, digits) for numberImg in resizedNumberImgs
    ]

    return tuple(actualNumbers)

def _classifyRectsDirectional(
    cropped: Any,
    rectGroups: List[List[tuple]],
    digits: Dict[int, Any],
    arrows: Dict[str, Any]
) -> Tuple[List[List[int]], List[str]]:
    '''The full algorithm to take in an area containing multiple building
    numbers with arrows, the bounding boxes of the numbers, and a digit and
    arrow classificaiton map, and return the true digits and arrow directions.
    '''

    # Pad the rects
    numberRectGroupsPadded = [[
        _padRect(pp, pp[3] * 0.08, pp[3] * 0.08) for pp in p
    ] for p in rectGroups]

    # Takes in three number rects and gets the rect of the adjacent arrow
    def arrowBox(p: tuple) -> tuple:
        a, _, c = p
        floatRect = (
            c[0] + c[2] + (mean(pp[2] for pp in p)) * 0.3,
            mean(pp[1] for pp in p),
            (c[0] - a[0]) * 0.6,
            mean(pp[2] for pp in p) * 1.4
        )
        return tuple(round(pp) for pp in floatRect)

    # Add the arrow box to the groups
    numberRectGroupsPadded = [
        (p[0], p[1], p[2], arrowBox(p)) for p in numberRectGroupsPadded
    ]
    
    # Defined by size of images in digits directory
    minW, minH = 28, 40
    templateRatio = minW / minH

    # Initialise lists for output
    actualNumbersGroup = []
    directionsGroup = []

    # Allows us to ignore duplicates
    visited = []

    # Iterate through groups
    for numberRectsPadded in numberRectGroupsPadded:
        # Initialise inner padded rects list
        newRects = []

        # Only proceed if not a duplicate (x, y) coord
        if numberRectsPadded[0][:2] not in visited:
            visited.append(numberRectsPadded[0][:2])

            # For each number/arrow rect, pad to match the template aspect ratio
            for numberRect in numberRectsPadded:
                _, _, w, h = numberRect
                ratio = w / h

                if ratio != templateRatio:
                    if ratio < templateRatio:
                        padX = (h * templateRatio) - w
                        newRects.append(_padRect(numberRect, padX, 0))
                    else:
                        padY = (w / templateRatio) - h
                        newRects.append(_padRect(numberRect, 0, padY))
                else:
                    newRects.append(numberRect)
                
            # Use the rects to crop out the number/arrow images
            numberImgs = [
                _cropImg(cropped, _rectToRotRect(rect)) for rect in newRects
            ]

            # Scale to match the template image size
            resizedNumberImgs = [
                cv.resize(numImg, (minW, minH)) for numImg in numberImgs
            ]

            # Classify numbers
            actualNumbers = [
                _matchNum(numImg, digits) for numImg in resizedNumberImgs[:-1]
            ]
            actualNumbers = actualNumbers if actualNumbers[1] != 6 else (
                actualNumbers[0], 0, actualNumbers[2]
            )
            
            # Classify arrow direction
            direction = _matchNum(resizedNumberImgs[-1], arrows)
            directionStr = 'left' if direction == 'l' else 'right'

            # Add to group output list if not a duplicate
            if actualNumbers not in actualNumbersGroup:
                actualNumbersGroup.append(actualNumbers)
                directionsGroup.append(directionStr)
    
    return actualNumbersGroup, directionsGroup

def _matchNum(
    img: Any,
    digits: Dict[Union[int, str], Any]
) -> Union[int, str]:
    '''Takes in a source image and map from desired values to lists of image
    templates, and uses it to classify the source image as one of the digits.'''

    # matchTemplate technique
    method = cv.TM_CCOEFF_NORMED

    # Initialise map of max values from matchTemplate
    maxima = {value: 0 for value in digits.keys()}

    img = cv.cvtColor(img.copy(), cv.COLOR_BGR2GRAY)

    # Iterate through digits and match the template, populating maxima
    # with the result
    for key, templates in digits.items():
        maxMatch = 0

        for template in templates:
            template = cv.cvtColor(template, cv.COLOR_BGR2GRAY)

            matchImg = img.copy()
            matchRes = cv.matchTemplate(matchImg, template, method)
            _, resValue, _, _ = cv.minMaxLoc(matchRes)

            maxMatch = max(maxMatch, resValue)
        
        maxima[key] = maxMatch
    
    # Find the best matching template
    maxKey, _ = max(maxima.items(), key=lambda item: item[1])

    return maxKey
  
# Utilities --------------------------------------------------------------------

def _aspectRatio(contour: Any) -> float:
    '''Computes the aspect ratio of a contour's bounding rectange
    (width / height).'''
    _, _, width, height = cv.boundingRect(contour)

    return width / height if height != 0 else 0

def _findEdges(img: Any, gaussian=3, t1=300, t2=400) -> Any:
    '''Applys the Canny edge detector to an image with preset values and
    preprocessing.'''
    blurred = cv.GaussianBlur(img, (gaussian, gaussian), 0)
    greyBlurred = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(greyBlurred, t1, t2)

    return edges

def _relativeSize(img: Any, contour: Any) -> float:
    '''Computes the relative size of a contour as a proportion of the source
    image size.'''
    imgWidth, imgHeight = img.shape[:2]
    _, _, cntWidth, cntHeight = cv.boundingRect(contour)

    imgArea = imgWidth * imgHeight
    contourArea = cntWidth * cntHeight

    return contourArea / imgArea if imgArea != 0 else 0

def _avgDiff(iterable: Iterable, key: LambdaType) -> float:
    '''Finds the average differences between elements in an iterable, given 
    a mapping function'''
    total = sum([key(x[0], x[1]) for x in permutations(iterable, 2)])
    numElements = len(list(permutations(iterable, 2)))

    return total / numElements

def _cropImg(img: Any, rotRect: Tuple[tuple, tuple, float]) -> Any:
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

def _numberAngle(numbers: Tuple[tuple, tuple, tuple]) -> float:
    '''Takes in a tuple of three bounding rects and finds the angle they slant
    at.'''
    angleOpp = ((numbers[2][0] + numbers[2][2]) - numbers[0][0])
    angleAdj = (
        (numbers[2][1] + numbers[2][3]) - (numbers[0][1] + numbers[0][3])
    )
    
    return np.arctan(angleOpp / angleAdj) if angleAdj != 0 else 0

def _padRect(
    rect: Tuple[int, int, int, int],
    padX: Union[int, float],
    padY: Union[int, float]
) -> Tuple[int, int, int, int]:
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
      
def _rectToRotRect(
    rect: Tuple[int, int, int, int]
) -> Tuple[Tuple[float, float], Tuple[float, float], float]:
    '''Converts a rect to a rotated rect'''
    x, y, w, h = rect
    return (x+(w/2), y+(h/2)), (w, h), 0

import cv2 as cv
import numpy as np
import os
from matplotlib import pyplot as plt

# Get the current project directory path
path = f'{os.path.dirname(os.path.abspath(__file__))}/'

def showImage(img, name):
    cv.imshow(name, img)
    cv.waitKey(0)
    cv.destroyAllWindows()


img1 = cv.imread(f'{path}prac03ex01img01.png')
img2 = cv.imread(f'{path}prac03ex01img02.png')
img3 = cv.imread(f'{path}prac03ex01img03.png')

# Harris corner detection
for (i, img) in enumerate([img1, img2, img3]):
    grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Apply harris algorithm to greyscale image
    harrisCorners = cv.cornerHarris(
        src = np.float32(grey),
        blockSize = 2,
        ksize = 3,
        k = 0.04,
    )

    # Dilate the result (Optional)
    harrisCorners = cv.dilate(harrisCorners, None)

    # Apply to original image with thresholding
    img[harrisCorners > 0.01 * harrisCorners.max()] = [0, 0, 255]

    showImage(img, f'Harris corners for image {i + 1}')

img1 = cv.imread(f'{path}prac03ex01img01.png')
img2 = cv.imread(f'{path}prac03ex01img02.png')
img3 = cv.imread(f'{path}prac03ex01img03.png')

# Shi-Tomasi corner detector
for (i, img) in enumerate([img1, img2, img3]):
    grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # Apply Shit-Tomasi to greyscale image
    corners = cv.goodFeaturesToTrack(
        image=grey,
        maxCorners=23,
        qualityLevel=0.01,
        minDistance=10,
        corners=None,
        blockSize=3,
        useHarrisDetector=False,
        k=0.04
    )

    # Draw circles on the source image
    for i in range(corners.shape[0]):
        cv.circle(
            img=img,
            center=(corners[i, 0, 0], corners[i, 0, 1]),
            radius=4,
            color=(120, 80, 200),
            thickness=cv.FILLED
        )
    
    showImage(img, f'Shi-Tomasi corners for image {i + 1}')

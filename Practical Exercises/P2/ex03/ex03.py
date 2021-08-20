import cv2 as cv
import numpy as np
import os

# Get the current project directory path
path = f'{os.path.dirname(os.path.abspath(__file__))}/'

def showImage(img, name):
    cv.imshow(name, img)
    cv.waitKey(0)
    cv.destroyAllWindows()

img = cv.imread(f'{path}prac02ex03img01.jpg')

# Seems to be the lowest value that leaves no noise
ksize = 5

imgMedian = cv.medianBlur(img, ksize)

showImage(imgMedian, f'Blurred by {ksize}')
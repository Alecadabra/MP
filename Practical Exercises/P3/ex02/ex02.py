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


imgChecker = ('Checker board', cv.imread(f'{path}prac03ex01img01.png'))
imgBoard = ('Car', cv.imread(f'{path}prac03ex02img01.jpg'))

# Canny edge detection
for (name, img) in (imgChecker, imgBoard):
    pass
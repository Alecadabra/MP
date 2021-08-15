import cv2 as cv
import numpy as np

relPath = 'Practical Exercises/P1/ex02/'

with open(f'{relPath}prac01ex02crop.txt', 'r') as file:
    # ['339', '341', '451', '378']
    cropStrings = file.readlines()[0].split(' ')

# 339, 341, 451, 378
x_l, y_l, x_r, y_r = [int(s) for s in cropStrings]

img = cv.imread(f'{relPath}prac01ex02img01.png')

croppedImg = img[y_l:y_r, x_l:x_r]
cv.imwrite(f'{relPath}prac01ex02img01cropped.png', croppedImg)

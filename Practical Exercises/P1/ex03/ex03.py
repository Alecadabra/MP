import cv2 as cv
import numpy as np

relPath = 'Practical Exercises/P1/ex03/'

with open(f'{relPath}prac01ex02crop.txt', 'r') as file:
    # ['339', '341', '451', '378']
    cropStrings = file.readlines()[0].split(' ')

# 339, 341, 451, 378
x_l, y_l, x_r, y_r = [int(s) for s in cropStrings]

img = cv.imread(f'{relPath}prac01ex02img01.png')

lineColour = (0x94, 0xd6, 0xbd) # RGB for cv.rectangle
pointColour = (0xa8, 0x53, 0x32) # BGR for cv.circle
imgBoxed = cv.rectangle(img, (x_l, y_l), (x_r, y_r), lineColour, 4)
for (x, y) in zip((x_l, x_l, x_r, x_r), (y_l, y_r) * 2):
    imgBoxed = cv.circle(imgBoxed, (x, y), 0, pointColour, 10)

cv.imwrite(f'{relPath}prac01ex03img01boxed.png', imgBoxed)

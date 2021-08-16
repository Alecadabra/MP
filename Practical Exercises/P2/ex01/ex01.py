import cv2 as cv
import os

# Get the current project directory path
path = f'{os.path.dirname(os.path.abspath(__file__))}/'

img = cv.imread(f'{path}prac02ex01img01.jpg')

colours = [
    ('Grey', cv.COLOR_BGR2GRAY),
    ('HSV',cv.COLOR_BGR2HSV),
    ('Luv', cv.COLOR_BGR2LUV),
    ('Lab', cv.COLOR_BGR2LAB)
]

for (name, code) in colours:
    converted = cv.cvtColor(img, code)
    cv.imshow(name, converted)
    cv.waitKey(0)
    cv.destroyAllWindows()

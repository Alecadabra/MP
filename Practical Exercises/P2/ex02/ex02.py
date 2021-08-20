import cv2 as cv
import numpy as np
import os

# Get the current project directory path
path = f'{os.path.dirname(os.path.abspath(__file__))}/'

def showImage(img, name):
    cv.imshow(name, img)
    cv.waitKey(0)
    cv.destroyAllWindows()

img1 = cv.imread(f'{path}prac02ex02img01.jpg')
img2 = cv.imread(f'{path}prac02ex02img02.jpg')

for (i, img) in enumerate([img1, img2]):
    showImage(img, f'No effects: image {i + 1}')

# Prewit kernels

kernelPrewitX = np.matrix([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
])
kernelPrewitY = np.matrix([
    [-1, -1, -1],
    [0, 0, 0],
    [1, 1, 1]
])

for (i, img) in enumerate([img1, img2]):
    imgPrewit = cv.filter2D(img, -1, kernelPrewitX)
    imgPrewit = cv.filter2D(imgPrewit, -1, kernelPrewitY)

    showImage(imgPrewit, f'Prewit kernel for image {i + 1}')

# Sobel kernels

kernelSobelX = np.matrix([
    [-1,  0,  1],
    [-2,  0,  2],
    [-1,  0,  1]
])
kernelSobelY = np.matrix([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]    
])

for (i, img) in enumerate([img1, img2]):
    imgSobel = cv.filter2D(img, -1, kernelSobelX)
    imgSobel = cv.filter2D(imgSobel, -1, kernelSobelY)

    showImage(imgSobel, f'Sobel kernel for image {i + 1}')

# Laplacian kernel

kernelLaplacian = np.matrix([
    [ 0,  1,  0],
    [ 1, -4,  1],
    [ 0,  1,  0]
])

for (i, img) in enumerate([img1, img2]):
    imgLaplacian = cv.filter2D(img, -1, kernelLaplacian)

    showImage(imgSobel, f'Laplacian kernel for image {i + 1}')

# Gaussian kernel (sigma = 1)

kernelGaussian = np.matrix([
    [ 1,  4,  7,  4,  1],
    [ 4, 16, 26, 16,  4],
    [ 7, 26, 41, 26,  7],
    [ 4, 16, 26, 16,  4],
    [ 1,  4,  7,  4,  1]
]) * (1/273)

for (i, img) in enumerate([img1, img2]):
    imgGaussian = cv.filter2D(img, -1, kernelGaussian)

    showImage(imgGaussian, f'Gaussian kernel for image {i + 1}')

# OpenCV GaussianBlur (sigma = 1)

for (i, img) in enumerate([img1, img2]):
    imgGaussian = cv.GaussianBlur(img, (5, 5), 1)

    showImage(imgGaussian, f'OpenCV GaussianBlur for image {i + 1}')

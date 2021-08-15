import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import os

# Get the current project directory path
path = f'{os.path.dirname(os.path.abspath(__file__))}/'

# Every image file path
imgPaths = [f'{path}prac01ex01img{num:02d}.png' for num in range(1,5)]

for (i, imgPath) in enumerate(imgPaths):
    
    shortImgPath = imgPath[(len(path)):]
    print(f'Reading in {shortImgPath}')

    img = cv.imread(imgPath, cv.IMREAD_COLOR)

    # Print out the image file name and its dimensions (width and height)

    width, height = img.shape[:-1]
    print(f'{shortImgPath} dimensions: Width = {width}, height = {height}')

    # Compute and print out the histogram (with 10 uniform bins) of each colour
    # channel (R, G, B). Make an observation of the histograms.

    colors = ('b', 'g', 'r')
    channels = cv.split(img)
    plt.figure()
    plt.title(f'Image {(i + 1):02d} Histogram')
    plt.xlabel('Bins')
    plt.ylabel('No. of pixels')
    img.ravel()

    print('Showing image histogram')

    for ((j, color), channel) in zip(enumerate(colors), channels):
        # Plot the current channel's histogram
        hist = cv.calcHist(
            images   = [img],
            channels = [j],
            mask     = None,
            histSize = [10],
            ranges   = [0, 256]
        )
        plt.plot(hist, color = color)
        plt.xlim([0, 10])
    plt.show()
    plt.close()

    # Reduce the size of the input image by 50% and output this as an image
    # file of the same format.

    print(r'Saving version of image reduced to 50% size')

    resized = cv.resize(img, (width // 2, height // 2))
    resizedPath = f'{imgPath[0:-4]}resized.png'
    cv.imwrite(resizedPath, resized)

    

import cv2
import numpy as np
from matplotlib import pyplot as plt

# Every image file path
relPath = 'Practical Exercises/P1/ex01/'
imgPaths = [f'{relPath}prac01ex01img{num:02d}.png' for num in range(1,5)]
imgPaths.append('Practical Exercises/P1/prac01ex04img01.png')

for (i, imgPath) in enumerate(imgPaths):
    img = cv2.imread(imgPath, cv2.IMREAD_COLOR)
    

    # Print out the image file name and its dimensions (width and height)

    width, height, _ = img.shape
    print(f'{imgPath}: Width = {width}, height = {height}')

    # Compute and print out the histogram (with 10 uniform bins) of each colour
    # channel (R, G, B). Make an observation of the histograms.

    colors = ('b', 'g', 'r')
    channels = cv2.split(img)
    plt.figure()
    plt.title(f'Image {(i + 1):02d} Histogram')
    plt.xlabel('Bins')
    plt.ylabel('No. of pixels')
    img.ravel()

    for ((j, color), channel) in zip(enumerate(colors), channels):
        # Plot the current channel's histogram
        hist = cv2.calcHist(
            images   = [img],
            channels = [j],
            mask     = None,
            histSize = [10],
            ranges   = [0, 256]
        )
        plt.plot(hist, color = color)
        plt.xlim([0, 10])
    cv2.imshow(imgPath, img)
    plt.show()
    cv2.destroyAllWindows()
    plt.close()

    #for (j, colour) in enumerate(colours):
        #hist = cv2.calcHist(
        #    images   = [img],
        #    channels = [j],
        #    mask     = None,
        #    histSize = [256],
        #    ranges   = [0, 256]
        #)
        #plt.plot(hist, color = colour)
        #plt.xlim([0, 256])
    #plt.show()
    

    # Reduce the size of the input image by 50% and output this as an image
    # file of the same format.

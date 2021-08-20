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

def plotImage(img, name):
    imgFlat = img.flatten()
    hist, = np.histogram(imgFlat, 256, [0, 256])[:-1]
    cdf = hist.cumsum()
    cdfNormalised = cdf * hist.max() / cdf.max()

    plt.figure(name)
    plt.plot(cdfNormalised, color = 'b')
    plt.hist(imgFlat, 256, [0, 256], color = 'r')
    plt.xlim([0, 256])
    plt.legend(('cdf', 'histogram'), loc = 'upper left')
    plt.show()


img = cv.imread(f'{path}prac02ex05img01.png', cv.IMREAD_GRAYSCALE)

showImage(img, 'Original')

plotImage(img, 'Original Histogram')

equalised = cv.equalizeHist(img)

showImage(equalised, 'Equalised')

plotImage(equalised, 'Equalised Histogram')

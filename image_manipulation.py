import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def imgDisplay(img):
    """Plot a given image on screen."""

    cv2.imshow('imgDisplay()', img)
    cv2.waitKey(0)
    cv2.destroyWindow('imgDisplay()')


def imgColorMask(img, color_boundary):
    """Return the given image masked to in a specific HSL color boundary"""

    # Convert to HSL color space ands separate components
    img_hsl = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

    # Correct the format of the color boundary
    lower_bound = np.array(color_boundary[0], dtype='uint8')
    upper_bound = np.array(color_boundary[1], dtype='uint8')

    # Create the color mask
    mask = cv2.inRange(img_hsl, lower_bound, upper_bound)

    return mask


def plotHistogram(img, n_channels=3):
    """Plot the histogram of every channel of a given image."""

    plot_histogram = plt.figure()

    # Plot the original image
    plt.subplot(1, 2, 1)
    plt.title('Original image')
    plt.imshow(img)

    # Plot the histogram for each channel
    for i in range(0, n_channels):
        plt.subplot(n_channels, 2, (i + 1) * 2)
        if (n_channels == 1):
            hist = cv2.calcHist([img[:, :]], [0], None, [256], [0, 256])
        else:
            hist = cv2.calcHist([img[:, :, i]], [0], None, [256], [0, 256])
        plt.title('Hist ' + str(i))
        plt.plot(hist)

    plt.show()

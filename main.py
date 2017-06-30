
# Third-party libraries
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Local modules
from image_manipulation import imgDisplay, imgColorMask, plotHistogram


def main():

    # Folder with the input image set
    filepath = 'dataset/f00002.png'

    # Load the image
    img = cv2.imread(filepath, cv2.IMREAD_COLOR)
    if img is None:
        print('Image couldn\'t be loaded.')

    # Define the color boundaries for the lane lines
    hsl_boundaries = [
        ([0, 0, 70], [180, 255, 255]),      # Yellow lines boundary
        ([0, 110, 10], [180, 255, 255])]    # White lines boundary

    # Apply a Gaussian blur filter to remove noise
    kernel_size = (3, 3)
    kernel = np.ones(kernel_size, np.uint8)
    img = cv2.GaussianBlur(img, kernel_size, 0)

    # Obtain the color masks for the lines segmentation
    yellow_mask = imgColorMask(img, hsl_boundaries[0])
    white_mask = imgColorMask(img, hsl_boundaries[1])

    # Apply an opening to the obtained masks
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)

    # cv2.imshow('Yellow Mask', yellow_mask)
    # cv2.imshow('White Mask', white_mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Plot the histograms for the HSL channels
    # plotHistogram(white_mask, n_channels=1)


if __name__ == '__main__':
    main()

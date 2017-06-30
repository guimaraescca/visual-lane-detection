import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


lane_map = np.float32([[250, 200], [356, 200], [0, 300], [495, 300]])
inverse_map = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])


def transform(img):
	"""Return the perspective transformation for the given image.

	Perform a perspective transformation to obtain a top-view image of the road
	for the given input. It improves the detection and approximation of the lane
	markings.
	"""

	trans_matrix = cv2.getPerspectiveTransform(lane_map, inverse_map)
	out_img = cv2.warpPerspective(img, trans_matrix, (300, 300))

	return out_img


def invTransform(img):
	"""Return the inverse perspective transformation for the given image"""

	inv_trans_matrix = cv2.getPerspectiveTransform(inverse_map, lane_map)
	out_img = cv2.warpPerspective(img, inv_trans_matrix, (640, 480))

	return out_img


def plot(img1, img2, img3):
	"""Plots the obtained images for the inverse mapping pipeline

	Usage: plot(img, img_map, img_inv_map)
	"""

	plt.subplot(131),
	plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
	plt.title('Input image')

	plt.subplot(132),
	plt.imshow(img2)
	plt.title('Perspective transformation')

	plt.subplot(133),
	plt.imshow(img3)
	plt.title('Inverse mapping')

	plt.show()

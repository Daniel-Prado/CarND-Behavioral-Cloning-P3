#PRUEBAS PREPROCESO

import csv
import numpy as np
import cv2


def perspective_transform(img, resize=(320,160)):
	pts1 = np.float32([[90,65],[230,65],[-160,160],[480,160]])
	pts2 = np.float32([[0,0],[320,0],[0,240],[320,240]])
	M = cv2.getPerspectiveTransform(pts1,pts2)
	dst = cv2.warpPerspective(img,M,(320,160))
	dst = cv2.resize(dst,resize, interpolation = cv2.INTER_CUBIC)
	return dst

def normalize_mean_std(img, individual_channels=True):
	img = img.astype('float64')
	if individual_channels == True:
		for channel in range(3):
			img[:,:,channel] -= np.mean(img[:,:,channel])
			img[:,:,channel] /= np.std(img[:,:,channel])
	else:
		img -= np.mean(img)
		img /= np.std(img)
	return img

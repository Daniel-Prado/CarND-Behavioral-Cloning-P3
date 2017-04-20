
import csv
import numpy as np
import cv2
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from preprocess import perspective_transform, normalize_mean_std

###########################################################
# Variables Definition
###########################################################
usecams = 'LCR' # 'C' for Center or 'LCR' for Left-Center-Right
correction = [0.0, 0.25, -0.25] # [C, L, R] corrections
DROP_PROB = 0.35
N_MULTIPLY = 4

cnn_resizing = (96,96)
cnn_input_shape = [96, 96, 3]


def shift_image(image,input_angle,max_range):
	''' Horizontal and vertical Translation
	Apply random horizontal translation to simulate car at various positions in the road
	For each pixel translation apply corresponding steering angle shift
	Minimum Horizontal shift is 5 pixels
	'''

	x_shift = np.random.uniform(low=5.0, high=max_range/2.0)
	#flip a coin to decide sign
	if np.random.randint(2) == 1:
		x_shift = x_shift * -1.0

	output_angle = input_angle + x_shift/max_range*0.4

	y_shift = np.random.uniform(-10,10)

	rows, cols = image.shape[:2]
	M = np.float32([[1,0,x_shift],[0,1,y_shift]])
	shifted_image = cv2.warpAffine(image,M,(cols,rows))

	return shifted_image,output_angle

def transf_brightness(img):
	'''
	image "brightness" (S channel in HSV) darkened between 0.1 (dark) and 1. (unchanged)
	img : input image in RGB format
	'''
	hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
	alpha = np.random.uniform(low=0.1, high=1.0, size=None)
	v = hsv[:,:,2]
	v = v * alpha
	v = np.clip(v,0,255)
	hsv[:,:,2] = v.astype('uint8')

	darker_image = cv2.cvtColor(hsv.astype('uint8'), cv2.COLOR_HSV2RGB)
	return darker_image

def resized(img, resize=None):
	if resize == None:
		resize = cnn_resizing
	return cv2.resize(img, resize, interpolation = cv2.INTER_CUBIC)

def cropped(img, high=65, low=20 ):
	return image[high:-low,:,:]

def load_csv_file():
	'''
	Opens the driving_log.csv file and loads its contents in a list of lines.
	'''
	lines = []
	with open('../data/driving_log.csv') as csvfile:
		reader = csv.reader(csvfile)
		for line in reader:
			lines.append(line)
	return lines	

def load_images(lines, usecams='LCR'):
	images = []
	measurements = []

	for line_items in lines:
		if usecams == 'LCR':
			img_line_range = 3
		elif usecams == 'C':
			img_line_range = 1
		else:
			print("DEBUG: invalid usecams value")
			return False

		# For each center, left and right camera images:
		for i in range(img_line_range):
			filename = '../data/IMG/' + line_items[i].split('/')[-1]
			image = cv2.imread(filename)

			images.append(image)
			# Read steering angle
			measurement = float(line_items[3])
			# Apply steering angle correction for center, left and right camera
			angle = measurement+correction[i]
			angles.append(angle)
	return True	

def augment_images(images, angles):
	augmented_images, augmented_angles = [], []
	for image, angle in zip(images,angles):
		# We crop right away:
		image = cropped(image)
		image = cropped(image)
		if abs(angle)<0.01:
			# We will take only 10% of the 0-angle images.
			if np.random.uniform(0.0, 1.0) > 0.9:
					augmented_images.append(resized(image))
					augmented_angles.append(angle)
		else:
			# First we include the original image, resized
			augmented_images.append(resized(image))
			augmented_angles.append(angle)
			# And also its flipped version
			augmented_images.append(resized(cv2.flip(image,1)))
			augmented_angles.append(angle*-1.0)

			# Now we obtain N_MULTIPLY augmented images of the original, applying x,y shift and 
			# randomly shifted image in x,y
			for _ in range(N_MULTIPLY):
				shifted_image, shifted_angle = shift_image(image, angle, 100)
				darkened_si = transf_brightness(shifted_image)

				augmented_images.append(resized(darkened_si))
				augmented_angles.append(shifted_angle)

				augmented_images.append(resized(cv2.flip(darkened_si,1)))
				augmented_angles.append(angle*-1.0)
	return augmented_images, augmented_angles


def main(_):

	lines = load_csv_file()
	images, angles = load_images(lines, 'LCR')
	augmented_images, augmented_angles = augment_images(images, angles)
	
	X_train = np.array(augmented_images)
	y_train = np.array(augmented_angles)

	### NVIDIA Model
	model = Sequential()
	model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=cnn_input_shape))
	#model.add(Cropping2D(cropping=((70,25),(0,0))))
	model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
	model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
	model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
	model.add(Convolution2D(64,3,3,activation="relu"))
	model.add(Convolution2D(64,3,3,activation="relu"))
	model.add(Flatten())

	model.add(Dense(100))
	model.add(Activation("relu"))
	model.add(Dropout(DROP_PROB))

	model.add(Dense(50))
	model.add(Activation("relu"))
	model.add(Dropout(DROP_PROB))

	model.add(Dense(10))
	model.add(Activation("relu"))
	model.add(Dropout(DROP_PROB))

	model.add(Dense(1))
	### End of NVIDIA Model

	model.compile(loss='mse', optimizer='adam')
	#model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=3)

	# save model architecture and weights at the end of the training
	with open('model.json', 'w') as f:
		f.write( model.to_json() )

	model.save('model.h5')
	print("Training complete!")



if __name__ == '__main__':
	tf.app.run()
	#CleanUp
	K.clear_session()
	gc.collect()



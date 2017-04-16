import csv
import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

lines = []
with open("/Users/Daniel/TRAIN_CAR/driving_log.csv") as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

images = []
measurements = []

for line in lines[0:1]:
	#Get the central image path from column 0
	source_path = line[0]
	filename = source_path.split('/')[-1]
	current_path = "/Users/Daniel/TRAIN_CAR/IMG/" + filename
	image = cv2.imread(current_path)

	pts1 = np.float32([[90,65],[230,65],[-160,160],[480,160]])
	pts2 = np.float32([[0,0],[320,0],[0,240],[320,240]])
	M = cv2.getPerspectiveTransform(pts1,pts2)
	dst = cv2.warpPerspective(image,M,(320,160))
	image = dst
	image = cv2.resize(image,(120,120), interpolation = cv2.INTER_CUBIC)

	#image[0] = cv2.equalizeHist(image[0])
	#image[1] = cv2.equalizeHist(image[1])
	#image[2] = cv2.equalizeHist(image[2])

	#image = image.astype('float64')

	#for channel in range(3):
	#	image[:,:,channel] -= np.mean(image[:,:,channel])
    #	image[:,:,channel] /= np.std(image[:,:,channel])

	images.append(image)
	#Get the steering value from column 3
	measurement = float(line[3]) 
	measurements.append(measurement)

augmented_images, augmented_measurements = [], []
for image, measurement in zip(images,measurements):
	augmented_images.append(image)
	augmented_measurements.append(measurement)
	augmented_images.append(cv2.flip(image,1))
	augmented_measurements.append(measurement*-1.0)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

###LENET Model
#model = Sequential()
#model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=[160,320,3]))
#model.add(Cropping2D(cropping=((70,25),(0,0))))
#model.add(Convolution2D(6,5,5,activation="relu"))
#model.add(MaxPooling2D())
#model.add(Convolution2D(6,5,5,activation="relu"))
#model.add(MaxPooling2D())
#model.add(Flatten())
#odel.add(Dense(120))
#model.add(Dense(84))
#model.add(Dense(1))
###End of LeNet model

from keras import backend as K

def perspectiveWarp(x):
#	x0 = Lambda(lambda x : x[:,:,:,0])(x)
#	x1 = Lambda(lambda x : x[:,:,:,1])(x)
#	x2 = Lambda(lambda x : x[:,:,:,2])(x)
	print(K.floatx(), "\n\n")
	sess  = K.get_session()
	x_array = sess.run(x)
	pts1 = np.float32([[90,65],[230,65],[-160,160],[480,160]])
	pts2 = np.float32([[0,0],[320,0],[0,240],[320,240]])
	M = cv2.getPerspectiveTransform(pts1,pts2)
	
	#x_array = cv2.warpPerspective(x,M,(320,160))
	return x



### NVIDIA Model
model = Sequential()
#model.add(Lambda(perspectiveWarp, input_shape=[320,160,3]))
#model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=[120,120,3]))
#model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu", input_shape=[120,120,3]))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
### End of NVIDIA Model



model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

model.save('model_NVidia_Perspective.h5')




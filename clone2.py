import csv
import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

lines = []
with open('../TRAIN_CAR/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

images = []
measurements = []

from preprocess import perspective_transform, normalize_mean_std

for line in lines:
	#Get the central image path from column 0
	source_path = line[0]
	filename = source_path.split('/')[-1]
	current_path = '../TRAIN_CAR/IMG/' + filename
	image = cv2.imread(current_path)

	#image = perspective_transform(image, (160,160))

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

DROP_PROB = 0.35

### NVIDIA Model
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=[160,320,3]))
model.add(Cropping2D(cropping=((70,25),(0,0))))
#model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu", input_shape=[160,160,3]))
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
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

model.save('model_data_NVidia_Relu_Drop035.h5')




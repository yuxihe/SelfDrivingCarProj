import csv
import cv2
import numpy as np
import os
import sklearn
import matplotlib.pyplot as plt

#read lines from driving_log.csv
samples = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
    # remove header
    samples.pop(0)

# split sample data set: 80% for training and 20% for validation
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# implementing a generator which can pull pieces of the data and process them on the fly when needed
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        # shuffle the data
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                # read in center, left and right images
                for i in range(3):
                    # find the image name
                    name = 'data/IMG/'+batch_sample[i].split('/')[-1]
                    originalImage = cv2.imread(name)
                    # change BGR to RGB
                    image = cv2.cvtColor(originalImage, cv2.COLOR_BGR2RGB)
                    # read in measurment
                    angle = float(batch_sample[3])
                    images.append(image)

                    # adjust the measurement according to the position of the camera: center--no change, left--plus 0.2, right--minus 0.2
                    if i == 0:
                        angles.append(angle)
                    elif i == 1:
                        angles.append(angle + 0.2)
                    else:
                        angles.append(angle - 0.2)
            
            # flip the image to generate more data points, also to reduce left or right turn bias
            augmented_images, augmented_angles = [], []

            for image, angle in zip(images, angles):
                augmented_images.append(image)
                augmented_angles.append(angle)
                # flip the image
                augmented_images.append(cv2.flip(image, 1))
                # take the measurement * -1.0
                augmented_angles.append(angle* -1.0)

            # convert image to np.array
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

ch, row, col = 3, 160, 320  # Trimmed image format

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, MaxPooling2D, Cropping2D

# the NVIDIA Autonomous Car Group model implementation
model = Sequential()
# trim image to only see section with road
model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(row,col,ch)))
# data normalization
model.add(Lambda(lambda x: x / 255.0 - 0.5))
model.add(Convolution2D(24,(5,5),strides=(2,2),activation="relu"))
model.add(Convolution2D(36,(5,5),strides=(2,2),activation="relu"))
model.add(Convolution2D(48,(5,5),strides=(2,2),activation="relu"))
model.add(Convolution2D(64,(3,3),activation="relu"))
model.add(Convolution2D(64,(3,3),activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
# add one output layer at the end to get only one output
model.add(Dense(1))

# config the learning process to use mean squared error as loss function and adam optimizer
model.compile(loss='mse', optimizer='adam')
# print out model summary for README.md file
model.summary()
# train the model with 5 epochs and test on validation set
history_object = model.fit_generator(train_generator,steps_per_epoch=(len(train_samples))/5,\
                                     validation_data=validation_generator,validation_steps=(len(validation_samples)/5),epochs=5,verbose=1)

# save the model
model.save('model.h5')

### print the keys contained in the history object
print(history_object.history.keys())
loss = history_object.history['loss']
valid_loss = history_object.history['val_loss']
print('Loss')
print(loss)
print('Validation Loss')
print(valid_loss)

### plot the training and validation loss for each epoch
#plt.plot(loss)
#plt.plot(valid_loss)
#plt.title('model mean squared error loss')
#plt.ylabel('mean squared error loss')
#plt.xlabel('epoch')
#plt.legend(['training set', 'validation set'], loc='upper right')
#plt.grid(color='black', linestyle='--', linewidth=1)
#plt.show()

exit()

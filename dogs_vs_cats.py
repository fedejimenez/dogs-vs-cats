#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 17:33:05 2019

@author: fede

"""

###########################################
# Convolutional Neural Network 

# Imstall Theano
# Install Tensorflow
# Instll Keras

#########################################

# PART 1 - Building the CNN

# Import the Keras libraries and packages
from keras.models import Sequential        # To initialize an NN as a sequence of layers
from keras.layers import Conv2D     # To use convolution layers (2D images)
from keras.layers import MaxPooling2D      # To use pooling layer
from keras.layers import Flatten           # To convert the the pool feature maps into an input vector
from keras.layers import Dense             # To add the fully connected layers into a classic ANN
 
# Initialize the CNN
classifier = Sequential()

# Step 1 - Convolutional layer
classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation = 'relu'))   # 32 feature detectors of 3x3 (rowxcol) | images: colored, 64x64

# Step 2 - Pooling to reduce the size of the feature map
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Add a second Convolutional Layer to improve accuracy
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))   
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening to prepare the input layer of a futur ANN (as a vector)
classifier.add(Flatten())   

# Step 4 - Full Connection (classic ANN)
classifier.add(Dense(activation = 'relu', units=128))            # 128 hidden nodes
classifier.add(Dense(activation = 'sigmoid', units=1))           # sigmoid for binary output: cat or dog
# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# PART 2 - Fitting the CNN to the images
# Image Augmentation - preprocess to prevent overfitting
from keras.preprocessing.image import ImageDataGenerator

# Prepare image augmentation
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

# Apply image augmentation to the training set and resizing images 
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

classifier.fit_generator(training_set,
                         steps_per_epoch=250,
                         epochs=25,
                         validation_data=test_set,
                         validation_steps=100
                         )

#################################################################
# PART 3 - Predict
# Step 1 - Save the model to disk
import pickle

filename = 'finalized_model.sav'
pickle.dump(classifier, open(filename, 'wb'))

# Serialize
with open('finalized_model.pkl', 'wb') as handle:
    pickle.dump(classifier, handle, pickle.HIGHEST_PROTOCOL)
    
# De-serialize
with open('finalized_model.pkl', 'rb') as handle:
    model = pickle.load(handle)    

# no we can call various methods over mlp_nn as as:    
from PIL import Image
import numpy as np
from skimage import transform

def load(filename):
   np_image = Image.open(filename)
   np_image = np.array(np_image).astype('float32')/255
   np_image = transform.resize(np_image, (64, 64, 3))
   np_image = np.expand_dims(np_image, axis=0)
   return np_image    

# Define classes
classes = training_set.class_indices  
print(classes)

# predict result
image = load('dataset/test_set/dogs/dog.4747.jpg')
type(image)

# Image from URL
from skimage import io

URL = 'https://www.dogster.com/wp-content/uploads/2018/09/Carolina-Dog.jpg'

img = io.imread(URL)
io.imshow(img)
io.show()

image = np.array(img).astype('float32')/255
image = transform.resize(image, (64, 64, 3))
image = np.expand_dims(image, axis=0)

# Start prediction
result = model.predict_classes(image)

if result[0][0] == 1:
    prediction = 'dog'
    accuracy = model.predict(image)[0][0]
else:
    prediction = 'cat'
    accuracy = 1 - model.predict(image)[0][0]


print('Result = ', prediction) 
print('Accuracy = ' + str(round(accuracy, 3)*100) + '%')

#########################################################

# Save model as h5
classifier.save('finalized_model.h5')

# Print weights
print(classifier.summary())
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
                         samples_per_epoch=8000,
                         epochs=25,
                         validation_data=test_set,
                         validation_steps=100
                         )
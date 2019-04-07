#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 16:25:20 2019

@author: Partha Pratim Sinha
"""

import tensorflow
import keras
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.models import Sequential
from keras.layers import Conv2D, SeparableConv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('/home/zeref/Human-Detector/Dataset/Train',target_size=(64, 64),batch_size=32,class_mode='binary')
test_set = test_datagen.flow_from_directory('/home/zeref/Human-Detector/Dataset/Test',target_size=(64, 64),batch_size=32,class_mode='binary')


classifier=Sequential()
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
classifier.add(MaxPooling2D())
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D())
classifier.add(SeparableConv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D())
classifier.add(Flatten())
classifier.add(Dense(128, activation='relu'))
classifier.add(Dense(1, activation='sigmoid'))

classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
classifier.fit_generator(training_set,steps_per_epoch=6562,epochs=10,validation_data=test_set,validation_steps=741)



test_image= image.load_img('/home/zeref/Human-Detector/1.png',target_size=(64,64))
test_image= image.img_to_array(test_image)
test_image=np.expand_dims(test_image,axis=0)
result=classifier.predict(test_image)
training_set.class_indices
if result[0][0]>=0.5:
    prediction='Human'
else:
    prediction='Non-Human'
prediction

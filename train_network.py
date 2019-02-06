# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 16:33:04 2019

@author: Marina
"""
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
from keras.utils import np_utils
import keras
from keras import backend as K

np.random.seed(1)
numberEpochs = 10
numberOutput = 10


def changeShapeData(visina,sirina, X_train, X_test):
        X_train = X_train.reshape(X_train.shape[0],visina,sirina,1)
        X_test = X_test.reshape(X_test.shape[0],visina,sirina,1)
        return X_train, X_test
    
def formatData(X_train, X_test):
            X_train = X_train.astype('float32')
            X_test = X_test.astype('float32')
            X_train = X_train/255
            X_test = X_test/255
            return X_train, X_test

def categorizeData(y_train, y_test):
            y_train=np_utils.to_categorical(y_train)
            y_test=np_utils.to_categorical(y_test)
            return y_train, y_test



(X_train, y_train), (X_test, y_test) = mnist.load_data()
sirina= X_train[0].shape[0]
visina = X_train[0].shape[1]
    
X_train, X_test = changeShapeData(visina,sirina, X_train, X_test)
X_train, X_test = formatData(X_train, X_test)
y_train, y_test = categorizeData(y_train, y_test)
 
#print(shape)
#print(X_train[0])
#print(y_train[0])
def addLayers(model,shape):
    
    model.add(Conv2D(filters = 32, kernel_size=(3, 3), input_shape=shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(filters=16, kernel_size=(3, 3)))
    model.add(Conv2D(filters=16, kernel_size=(3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(output_dim =128))
    model.add(Activation('relu'))
    model.add(Dense(output_dim =64))
    model.add(Activation('relu'))
    model.add(Dense(numberOutput))
    model.add(Activation('softmax'))
    
    
    return model
 
def initMetod():
    print(X_train[0].shape)
    print(keras.__version__)
    
    if K.image_data_format() == 'channels_first':
        shape = (1, sirina, visina)
    else:
        shape = (sirina, visina, 1)
    
    model = Sequential()
    model = addLayers(model,shape)
    print(model.summary())
    return model

def getModel():
    model=initMetod()
    model.load_weights("network.h5")
    return model

          
def trainModel():
        model=initMetod()
        model.compile(optimizer='adam',loss='categorical_crossentropy',  metrics=['accuracy'])
        history=model.fit(X_train, y_train, batch_size=128, epochs=numberEpochs, validation_split=0.25  )
        rez = model.evaluate(X_test, y_test, verbose=1)
        print("Accuracy: %.2f%%" % (rez[1]*100))
        model.save_weights("network.h5")

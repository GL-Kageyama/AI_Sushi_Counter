# Copyright (c) 2020 a1kageyama
# Released under the MIT license

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adamax


# Definition of the CNN model
def defModel(inShape, nbClass):
    model = Sequential()

    model.add(Conv2D(64, (3, 3), activation="relu", name="block1_conv", input_shape=inShape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="block1_pool"))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), activation="relu", name="block2_conv"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="block2_pool"))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, (3, 3), activation="relu", name="block3_conv"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="block3_pool"))
    model.add(Dropout(0.25))

    model.add(Conv2D(512, (3, 3), activation="relu", name="block4_conv"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="block4_pool"))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(4096, activation="relu", name="fc1"))
    model.add(Dropout(0.5, name="dense1_dropout"))
    model.add(Dense(4096, activation="relu", name="fc2"))
    model.add(Dropout(0.5, name="dense2_dropout"))

    model.add(Dense(nbClass, activation="softmax", name="predictions"))

    return model


# Getting a CNN Model
def getModel(inShape, nbClass):

    model = defModel(inShape, nbClass)

    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
        metrics=["accuracy"])

    return model

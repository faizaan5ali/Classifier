import os
import math
import random
import shutil
import scipy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

import tensorflow as tf
from keras import Model
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

tf.keras.mixed_precision.set_global_policy('mixed_float16')

BASE_DIR = 'data/'
OG_DATA_DIR = 'data/64/'
subfolders = next(os.walk(OG_DATA_DIR))[1]
print(subfolders)

if not os.path.isdir(BASE_DIR + 'train/'):
    for name in subfolders:
        os.makedirs(BASE_DIR + 'train/' + name)
        os.makedirs(BASE_DIR + 'val/' + name)
        os.makedirs(BASE_DIR + 'test/' + name)

if not os.path.isdir(BASE_DIR + 'test/'):
    for folder_num, folder in enumerate(subfolders):
        files = os.listdir(OG_DATA_DIR + folder + "/")
        num_images = len([name for name in files])
        train = int((num_images * 0.6) + 0.5)
        valid = int((num_images * 0.25) + 0.5)
        test = num_images - train - valid
        print(num_images, train, valid, test)
        for image_id, image in enumerate(files):
            image_name = OG_DATA_DIR + folder + "/" + image
            if image_id < train:
                shutil.move(image_name, BASE_DIR + "train/" + folder)
            elif image_id < train + valid:
                shutil.move(image_name, BASE_DIR + "val/" + folder)
            else:
                shutil.move(image_name, BASE_DIR + "test/" + folder)

train_gen = keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255, horizontal_flip=True, vertical_flip=True, rotation_range=90)
valid_gen = keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
test_gen = keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

target_size = (64, 64)

train_batches = train_gen.flow_from_directory(
    'data/train',
    target_size=target_size,
    class_mode='categorical',
    batch_size=128,
    shuffle=True,
    color_mode="grayscale",
    classes=subfolders
)

valid_batches = valid_gen.flow_from_directory(
    'data/val',
    target_size=target_size,
    class_mode='categorical',
    batch_size=128,
    shuffle=False,
    color_mode="grayscale",
    classes=subfolders
)

test_batches = test_gen.flow_from_directory(
    'data/test',
    target_size=target_size,
    class_mode='categorical',
    batch_size=16,
    shuffle=False,
    color_mode="grayscale",
    classes=subfolders
)


def show(batch, pred_labels=None):
    plt.figure(figsize=(10, 10))
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(batch[0][i], cmap=plt.cm.binary)
        lbl = subfolders[np.argmax(batch[1][i])]
        if pred_labels is not None:
            lbl += "/ Pred: " + subfolders[int(pred_labels[i])]
        plt.xlabel(lbl)
    plt.show()


# Function to dynamically adjust the learning rate
def lr_schedule(epoch, lr):
    if epoch < 11:
        return 0.0005
    else:
        return 0.005


# Function to dynamically adjust the momentum
def momentum_schedule(epoch):
    if epoch < 11:
        return 0.95
    else:
        return 0.85


sgd_optimizer = SGD(learning_rate=0.001, momentum=0.9)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=1e-6)

model = Sequential()
model.add(Conv2D(64, 3, input_shape=(64, 64, 1), activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(128, 3, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(256, 3, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(200, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=sgd_optimizer, metrics=['accuracy'])
history = model.fit(train_batches, validation_data=valid_batches, epochs=50, callbacks=[early_stopping, reduce_lr])

model.evaluate(test_batches)

model.save("lego_model_4layer_bw_v2.h5")

# plot loss and acc
plt.figure(figsize=(16, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='valid loss')
plt.grid()
plt.legend(fontsize=15)

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='valid acc')
plt.grid()
plt.legend(fontsize=15)

plt.show()

# make some predictions
predictions = model.predict(test_batches)
predictions = tf.nn.softmax(predictions)
labels = np.argmax(predictions, axis=1)

print(test_batches[0][1])
print(labels[0:4])
show(test_batches[0], labels[0:4])

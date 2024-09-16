import os
import numpy as np
from keras.utils import image_dataset_from_directory
from keras import layers
import tensorflow_datasets as tfds
from keras.models import Model

test_dir = 'D:\\111_NESYSTEM_PAPKI\\augmented\\test'

train = image_dataset_from_directory(directory='D:\\111_NESYSTEM_PAPKI\\augmented\\train', color_mode='grayscale',
                                     batch_size=len(os.listdir('D:\\111_NESYSTEM_PAPKI\\augmented\\train')),
                                     image_size=(540, 420),
                                     shuffle=False, interpolation='bilinear', labels=None, label_mode=None)

for ex in tfds.as_numpy(train):
    for_train = ex

for_train = for_train / 255

train_cleaned = image_dataset_from_directory(directory='D:\\111_NESYSTEM_PAPKI\\augmented\\train_cleaned',
                                             color_mode='grayscale',
                                             batch_size=len(os.listdir('D:\\111_NESYSTEM_PAPKI\\augmented\\train_cleaned')),
                                             image_size=(540, 420),
                                             shuffle=False, interpolation='bilinear', labels=None, label_mode=None)

for ex in tfds.as_numpy(train_cleaned):
    cleaned_for_train = ex

cleaned_for_train = cleaned_for_train / 255

valid = image_dataset_from_directory(directory='D:\\111_NESYSTEM_PAPKI\\augmented\\valid', color_mode='grayscale',
                                     batch_size=len(os.listdir('D:\\111_NESYSTEM_PAPKI\\augmented\\valid')),
                                     image_size=(540, 420),
                                     shuffle=False, interpolation='bilinear', labels=None, label_mode=None)

for ex in tfds.as_numpy(valid):
    for_valid = ex

for_valid = for_valid / 255

valid_cleaned = image_dataset_from_directory(directory='D:\\111_NESYSTEM_PAPKI\\augmented\\valid_cleaned',
                                             color_mode='grayscale',
                                             batch_size=len(os.listdir('D:\\111_NESYSTEM_PAPKI\\augmented\\valid_cleaned')),
                                             image_size=(540, 420),
                                             shuffle=False, interpolation='bilinear', labels=None, label_mode=None)

for ex in tfds.as_numpy(valid_cleaned):
    cleaned_for_valid = ex

cleaned_for_valid = cleaned_for_valid / 255


# encoder
input_img = layers.Input(shape=(540, 420, 1))

x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(input_img)
x = layers.MaxPooling2D((2, 2), padding="same")(x)
x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
x = layers.MaxPooling2D((2, 2), padding="same")(x)
encoded = layers.BatchNormalization()(x)

# Decoder
x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(encoded)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(16, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)
decoded = layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer="adam", loss="mean_squared_error")
autoencoder.summary()

autoencoder.fit(x=for_train, y=cleaned_for_train, epochs=30, batch_size=32,
                shuffle=False, validation_data=(for_valid, cleaned_for_valid))

autoencoder.save('autoencoder_4.h5')
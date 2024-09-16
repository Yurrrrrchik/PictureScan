import keras
import numpy as np
import os
import cv2
from keras.utils import image_dataset_from_directory
import tensorflow_datasets as tfds

model = keras.models.load_model('autoencoder_4.h5')


def count_metric(path, metric):
    for file in os.listdir(path):
        clear = cv2.imread(os.path.join(path + '_cleaned', file))
        dirty = cv2.imread(os.path.join(path, file))
        cv2.imshow('Original', dirty)
        cv2.imshow('Must-be', clear)
        clear = cv2.cvtColor(clear, cv2.COLOR_BGR2GRAY)
        dirty = cv2.cvtColor(dirty, cv2.COLOR_BGR2GRAY)
        shapes = (clear.shape[0], clear.shape[1])
        dirty = cv2.resize(dirty, (420, 540))
        dirty = dirty / 255
        clear = clear / 255
        dirty = np.expand_dims(dirty, axis=0)
        dirty = np.expand_dims(dirty, axis=3)
        predicted = model.predict(dirty)
        predicted = np.squeeze(predicted)
        predicted = cv2.resize(predicted, (shapes[1], shapes[0]))
        cv2.imshow('Predicted', predicted)
        cv2.waitKey(5)
        metric.update_state(predicted, clear)
    return float(metric.result())


mse = keras.metrics.MeanSquaredError()
mse_value = count_metric('D:\\111_NESYSTEM_PAPKI\\augmented\\test', mse)
print(f"MeanSquaredError metric: {mse_value}")

mae = keras.metrics.MeanAbsoluteError()
mae_value = count_metric('D:\\111_NESYSTEM_PAPKI\\augmented\\test', mae)
print(f"MeanAbsoluteError metric: {mae_value}")
# import keras
import cv2
import numpy as np
import os
from skimage.metrics import mean_squared_error

directory = 'D:\\111_NESYSTEM_PAPKI\\augmented\\test'

global_error = 0
counter = 0

def clean(picture):
    brightened_image = cv2.convertScaleAbs(picture, alpha=1.4, beta=0)

    # Повышение резкости
    sharpening_kernel = np.array([[0, -1, 0],
                                  [-1, 5.5, -1],
                                  [0, -1, 0]])
    sharpened_image = cv2.filter2D(brightened_image, -1, sharpening_kernel)

    gray_image = cv2.cvtColor(sharpened_image, cv2.COLOR_BGR2GRAY)

    # Enhance the image using CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(gray_image)
    # enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_GRAY2BGR)

    return enhanced_image

# def count_metric(directory):
#    for file in os.listdir(directory):
#        counter += 1
#        dirty = cv2.imread(os.path.join(directory, file))
#        clear = cv2.imread(os.path.join(directory + "_cleaned", file))
#        clear = cv2.cvtColor(clear, cv2.COLOR_BGR2GRAY)

#        clean(dirty)

#       a = keras.metrics.MeanSquaredError()
#        a.update_state(clear, enhanced_image)

#        cv2.imshow('Original Image', dirty)
#        cv2.imshow('Must-be', clear)
#        cv2.imshow('Enhanced Image', enhanced_image)
        # cv2.imshow('En Image', tr)
#        cv2.waitKey(1000)
        # cv2.destroyAllWindows()

#        return float(a.result())
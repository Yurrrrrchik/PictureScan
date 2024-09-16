from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import cv2
from keras.utils import image_dataset_from_directory, load_img, img_to_array
import random

augmentator = ImageDataGenerator(fill_mode='nearest', rescale=1. / 255, rotation_range=20,
                                 zoom_range=0.02, vertical_flip=True, horizontal_flip=True,
                                 width_shift_range=0.02, height_shift_range=0.02)

origin_directory = 'D:\\111_NESYSTEM_PAPKI\divided\\'
target_directory = 'D:\\111_NESYSTEM_PAPKI\\augmented\\'

seeeed = random.randrange(0, 500)
def augment(dataset_type):
    seed = seeeed
    original_number = 0
    if not dataset_type in os.listdir(target_directory):
        os.mkdir(os.path.join(target_directory, dataset_type))
    if not dataset_type + '_cleaned' in os.listdir(target_directory):
        os.mkdir(os.path.join(target_directory, dataset_type + '_cleaned'))
    for filename in os.listdir(os.path.join(origin_directory, dataset_type)):
        augmented_number = 0
        current_path = os.path.join(origin_directory, dataset_type)
        picture = load_img(os.path.join(current_path, filename))
        picture = img_to_array(picture)
        picture = picture.reshape((1,) + picture.shape)
        for epoch in augmentator.flow(picture, batch_size=1, save_to_dir=os.path.join(target_directory, dataset_type),
                                      save_prefix=f'c{original_number}' + filename.replace('.png', ''),
                                      save_format='png', seed=seed):
            augmented_number += 1
            original_number += 1
            if augmented_number == 10:
                break
        seed += 50
    seed = seeeed
    original_number = 0
    for filename in os.listdir(os.path.join(origin_directory, dataset_type + '_cleaned')):
        augmented_number = 0
        current_path = os.path.join(origin_directory, dataset_type + '_cleaned')
        picture = load_img(os.path.join(current_path, filename))
        picture = img_to_array(picture)
        picture = picture.reshape((1,) + picture.shape)
        for epoch in augmentator.flow(picture, batch_size=1, save_to_dir=os.path.join(target_directory,
                                    dataset_type + '_cleaned'), save_prefix=f'c{original_number}' +
                                    filename.replace('.png', ''), save_format='png', seed=seed):
            augmented_number += 1
            original_number += 1
            if augmented_number == 10:
                break
        seed += 50

augment('test')
augment('train')
augment('valid')
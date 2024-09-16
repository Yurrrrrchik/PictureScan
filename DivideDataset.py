import os
import shutil
import math
import cv2

train_coef = 0.70
valid_coef = 0.15
test_coef = 0.15

origin_directory = 'D:\\111_NESYSTEM_PAPKI\denoising-dirty-documents'
target_directory = 'D:\\111_NESYSTEM_PAPKI\divided'


for papka in os.listdir(origin_directory):
    if papka not in os.listdir(target_directory):
        os.mkdir(os.path.join(target_directory, papka))
    papka_path = os.path.join(origin_directory, papka)
    target_path = os.path.join(target_directory, papka)
    if papka != 'test':
        for file in os.listdir(papka_path)[:math.floor(len(os.listdir(papka_path)) * train_coef)]:
            original = os.path.join(papka_path, file)
            clone = os.path.join(target_path, file)
            shutil.copy2(original, clone)


if 'valid' not in os.listdir(target_directory):
    os.mkdir(os.path.join(target_directory, 'valid'))
if 'valid_cleaned' not in os.listdir(target_directory):
    os.mkdir(os.path.join(target_directory, 'valid_cleaned'))
for papka in os.listdir(origin_directory):
#    if 'valid' not in os.listdir(target_directory):
#        os.mkdir(os.path.join(target_directory, papka))
    papka_path = os.path.join(origin_directory, papka)
    target_path = os.path.join(target_directory, 'valid' if papka == 'train' else 'valid_cleaned')
    if papka == 'test':
        continue
    else:
        left = math.floor(len(os.listdir(papka_path)) * train_coef)
        right = math.floor(len(os.listdir(papka_path)) * train_coef) + round(len(os.listdir(papka_path)) * valid_coef)
        for file in os.listdir(papka_path)[left: right]:
            original = os.path.join(papka_path, file)
            clone = os.path.join(target_path, file)
            shutil.copy2(original, clone)


if 'test' not in os.listdir(target_directory):
    os.mkdir(os.path.join(target_directory, 'test'))
if 'test_cleaned' not in os.listdir(target_directory):
    os.mkdir(os.path.join(target_directory, 'test_cleaned'))
for papka in os.listdir(origin_directory):
#    if 'tst' not in os.listdir(target_directory):
#        os.mkdir(os.path.join(target_directory, papka))
    papka_path = os.path.join(origin_directory, papka)
    target_path = os.path.join(target_directory, 'test' if papka == 'train' else 'test_cleaned')
    if papka == 'test':
        continue
    else:
        left = math.floor(len(os.listdir(papka_path)) * train_coef) + round(len(os.listdir(papka_path)) * valid_coef)
        for file in os.listdir(papka_path)[left:]:
            original = os.path.join(papka_path, file)
            clone = os.path.join(target_path, file)
            shutil.copy2(original, clone)


for papka in os.listdir(target_directory):
    current_path = os.path.join(target_directory, papka)
    for file in os.listdir(current_path):
        file_path = os.path.join(current_path, file)
        img = cv2.imread(file_path)
        if img.shape == (540, 420):
            continue
        img = cv2.resize(img, (540, 420))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(file_path, img)


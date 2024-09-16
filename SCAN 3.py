import easyocr
import cv2
from tensorflow.keras.models import load_model
import numpy as np

reader = easyocr.Reader(['en'])

autoencoder = load_model('autoencoder_4.h5')

# Функция для обработки изображения с использованием автоэнкодера
def process_image(image):
    # Преобразование изображения в черно-белое
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Нормализация значений пикселей
    normalized_image = gray_image / 255.0
    # Изменение размерности изображения для соответствия входу автоэнкодера
    resized_image = cv2.resize(normalized_image, (420, 540))
    # Расширение размерности изображения для совместимости с моделью
    input_image = resized_image.reshape((1, 540, 420, 1))
    # Применение автоэнкодера к изображению
    processed_image = autoencoder.predict(input_image)
    # Извлечение обработанного изображения из прогноза
    output_image = processed_image[0, :, :, 0]
    output_image = cv2.resize(output_image, (image.shape[1], image.shape[0]))
    return output_image

input_image = cv2.imread('D:\\111_NESYSTEM_PAPKI\denoising-dirty-documents\\train\\192.png')

# Обработка изображения с использованием автоэнкодера
processed_image = process_image(input_image)

# Извлечение текста с обработанного изображения
result = reader.readtext(input_image)

# Вывод результатов
cv2.imshow('Original Image', input_image)
cv2.imshow('Processed Image', processed_image)
print('Extracted Text:', result)

cv2.waitKey(0)
cv2.destroyAllWindows()
import cv2
from tensorflow.keras.models import load_model
import numpy as np

# Загрузка предварительно обученной модели автоэнкодера
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

# Загрузка входного изображения
input_image = cv2.imread('D:\\111_NESYSTEM_PAPKI\denoising-dirty-documents\\test\\130.png')

# Обработка изображения с использованием автоэнкодера
processed_image = process_image(input_image)

def letters_extract(image_file: str, out_size=28) -> list[any]:
    img = cv2.imread(image_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    img_erode = cv2.erode(thresh, np.ones((3, 3), np.uint8), iterations=1)

    # Get contours
    contours, hierarchy = cv2.findContours(img_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    output = img.copy()

    letters = []
    for idx, contour in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(contour)
        # print("R", idx, x, y, w, h, cv2.contourArea(contour), hierarchy[0][idx])
        # hierarchy[i][0]: the index of the next contour of the same level
        # hierarchy[i][1]: the index of the previous contour of the same level
        # hierarchy[i][2]: the index of the first child
        # hierarchy[i][3]: the index of the parent
        if hierarchy[0][idx][3] == 0:
            cv2.rectangle(output, (x, y), (x + w, y + h), (70, 0, 0), 1)
            letter_crop = gray[y:y + h, x:x + w]
            # print(letter_crop.shape)

            # Resize letter canvas to square
            size_max = max(w, h)
            letter_square = 255 * np.ones(shape=[size_max, size_max], dtype=np.uint8)
            if w > h:
                # Enlarge image top-bottom
                # ------
                # ======
                # ------
                y_pos = size_max//2 - h//2
                letter_square[y_pos:y_pos + h, 0:w] = letter_crop
            elif w < h:
                # Enlarge image left-right
                # --||--
                x_pos = size_max//2 - w//2
                letter_square[0:h, x_pos:x_pos + w] = letter_crop
            else:
                letter_square = letter_crop

            # Resize letter to 28x28 and add letter and its X-coordinate
            letters.append((x, w, cv2.resize(letter_square, (out_size, out_size), interpolation=cv2.INTER_AREA)))

    # Sort array in place by X-coordinate
    letters.sort(key=lambda x: x[0], reverse=False)

    return letters

def emnist_predict_img(model, img):
    img_arr = np.expand_dims(img, axis=0)
    img_arr = 1 - img_arr/255.0
    img_arr[0] = np.rot90(img_arr[0], 3)
    img_arr[0] = np.fliplr(img_arr[0])
    img_arr = img_arr.reshape((1, 28, 28, 1))

    predict = autoencoder.predict([img_arr])
    result = np.argmax(predict, axis=1)
    return chr(emnist_labels[result[0]])

def img_to_str(model, image_file):
    letters = letters_extract(image_file)
    s_out = ""
    for i in range(len(letters)):
        dn = letters[i+1][0] - letters[i][0] - letters[i][1] if i < len(letters) - 1 else 0
        s_out += emnist_predict_img(model, letters[i][2])
        if (dn > letters[i][1]/4):
            s_out += ' '
    return s_out

s_out = img_to_str(autoencoder, 'D:\\111_NESYSTEM_PAPKI\denoising-dirty-documents\\test\\130.png')

# Вывод результатов
cv2.imshow('Original Image', input_image)
cv2.imshow('Processed Image', processed_image)
print('Extracted Text:', s_out)

cv2.waitKey(0)
cv2.destroyAllWindows()
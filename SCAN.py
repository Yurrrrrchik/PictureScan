import cv2
import pytesseract
from tensorflow.keras.models import load_model
import numpy as np

autoencoder = load_model('autoencoder_4.h5')

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def process_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    normalized_image = gray_image / 255.0
    resized_image = cv2.resize(normalized_image, (420, 540))
    input_image = resized_image.reshape((1, 540, 420, 1))
    processed_image = autoencoder.predict(input_image)
    output_image = processed_image[0, :, :, 0]
    output_image = cv2.resize(output_image, (image.shape[1], image.shape[0]))
    return output_image


def extract_text(image):
    image = cv2.convertScaleAbs(image)
    # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshold_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    text = pytesseract.image_to_string(image)
    return text

input_image = cv2.imread('D:\\111_NESYSTEM_PAPKI\denoising-dirty-documents\\train\\15.png')
processed_image = process_image(input_image)
extracted_text = extract_text(processed_image)

cv2.imshow('Original Image', input_image)
cv2.imshow('Processed Image', processed_image)
print('Extracted Text:', extracted_text)

cv2.waitKey(0)
cv2.destroyAllWindows()
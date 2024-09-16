import cv2
import easyocr
from UsingCV import clean
import numpy as np

reader = easyocr.Reader(['en'])

def scan(input_image):
    processed_image = clean(input_image)
    result = reader.readtext(processed_image)
    extracted_text = ''
    for box in result:
        extracted_text += box[1] + ' '  # Извлекаем текст из результата распознавания
    return processed_image, extracted_text.strip()

input_image = cv2.imread('D:\\111_NESYSTEM_PAPKI\Picturess\\po.png')
processed_image, extracted_text = scan(input_image)

cv2.imshow('Original Image', input_image)
cv2.imshow('Processed Image', processed_image)
print('Extracted Text:', extracted_text)

cv2.waitKey(0)
cv2.destroyAllWindows()
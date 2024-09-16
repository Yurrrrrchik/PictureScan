import cv2
import pytesseract
from UsingCV import clean
import numpy as np

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
tessdata_dir_config = r'--tessdata-dir "C:\Program Files (x86)\Tesseract-OCR\tessdata"'

def scan(input_image, lang='rus'):
    processed_image = clean(input_image)
    extracted_text = pytesseract.image_to_string(processed_image, lang)
    return processed_image, extracted_text

# input_image = cv2.imread('D:\\111_NESYSTEM_PAPKI\divided\\train\\134.png')
# processed_image, extracted_text = scan(input_image)

# cv2.imshow('Original Image', input_image)
# cv2.imshow('Processed Image', processed_image)
# print('Extracted Text:', extracted_text)

# cv2.waitKey(0)
# cv2.destroyAllWindows()
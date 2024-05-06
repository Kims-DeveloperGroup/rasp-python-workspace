from pprint import pprint

import cv2
import pytesseract

from picamera2 import MappedArray, Picamera2, Preview

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration({"size": (1024, 768)}))
picam2.start_preview(Preview.QTGL)
picam2.start()

threshold = 50


while True:
    data = picam2.capture_array()
    
    pprint(data, width=500, sort_dicts=False)
    cv2.imshow('rame', data)
import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation

segmentor = SelfiSegmentation()

# open camera 
cap = cv2.VideoCapture(1)

while True:
    # read image
    ret, img = cap.read()

    #resize office to 640×480
    img = cv2.resize(img, (320, 240))
    green = (0, 255, 0)
    imgNoBg = segmentor.removeBG(img, green, threshold=0.50)

    # show both images
    cv2.imshow('office',img)
    cv2.imshow('office no bg',imgNoBg)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# close camera
cap.release()
cv2.destroyAllWindows()

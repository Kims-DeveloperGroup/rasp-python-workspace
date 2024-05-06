from pprint import pprint
import cv2
from picamera2 import Picamera2, Preview
from picamera2.encoders import H264Encoder

#fourcc= cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter('/home/rica/Documents/ret/ret_nobg.avi', fourcc, 20.0, (640, 480))

# Configure Picamera2
picam2 = Picamera2()
#video_config = picam2.create_video_configuration()
capture_config = picam2.create_still_configuration()
picam2.configure(picam2.create_preview_configuration({"size": (640, 480)}))
#picam2.start_preview(Preview.QTGL)

picam2.configure(capture_config)# Without the config, Unsupported foramt error is thrown at fgbg.apply()
picam2.start()

# Remove Background
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

## cature frames untils pressing 'q' key
while True:
    frame = picam2.capture_array()
    pprint(frame, width=500, sort_dicts=False)
    #out.write(frame)
    frame_nobg = fgbg.apply(frame)
    cv2.imshow('frame_origin', frame)
    cv2.imshow('frame', frame_nobg)
    
    # Exit condition: 'q' pressed 
    key = cv2.waitKey(1)
    print(key)
    if key == ord('q'):
        break
picam2.stop()
#picam2.stop_preview)
##out.release()

print('Processing frames')
#cap = cv2.VideoCapture('/home/rica/Documents/ret/ret.h264')
#while True:
#    ret, frame = cap.read()
#    frame_nobg = fgbg.apply(frame)
#    cv2.imshow('No Background', frame_nobg)
#    if cv2.waitKey(10) == ord('q'):
#        break

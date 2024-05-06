import time
import picamera2 as picamera

camera = picamera.Picamera2()
try:
    camera.start_preview()
    time.sleep(10)
    camera.stop_preview()
finally:
    camera.close()
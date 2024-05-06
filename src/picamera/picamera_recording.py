import picamera2 as picamera
import time
from libcamera import controls

with picamera.Picamera2() as picam2:
    picam2.start(show_preview=True)
    time.sleep(5)
    #picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})
    picam2.start_and_capture_files("fastfocus{:d}.jpg", num_files=3, delay=0.5)
    picam2.stop_preview()
    picam2.stop()
from picamera2.encoders import H264Encoder
from picamera2 import Picamera2, Preview
import time
picam2 = Picamera2()
video_config = picam2.create_video_configuration()
picam2.configure(video_config)
encoder = H264Encoder(bitrate=10000000)
output = "/home/rica/Documents/ret/ret.h264"

# Wait 10 seconds before recording
picam2.start_preview(Preview.QT)
time.sleep(10)

#record for 10 seconds
picam2.start_recording(encoder, output)
time.sleep(10)
picam2.stop_preview()
picam2.stop_recording()

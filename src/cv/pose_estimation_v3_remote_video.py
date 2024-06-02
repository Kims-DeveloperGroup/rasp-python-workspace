from dataclasses ataclass
import cv2
from cv2 import VideoCapture
import matplotlib.pyplot as plt
import numpy as np
import mediapipe as mp
import csv

#import sys
#sys.path.append('../')
#from ...common.youtube import getVideoCapture
import pafy


def write_landmarks_to_csv(landmarks):
	csv_data = []
	for idx, landmark in enumerate(landmarks):
		print(f"{mp_pose.PoseLandmark(idx).name}: (x: {landmark.x}, y: {landmark.y}, z: {landmark.z})")
		csv_data.append([ idx, landmark.x, landmark.y, landmark.z])
	print("\n")
	return csv_data

# Init data.csv and open the file writer
data_file_path = '/Users/rica/Documents/data_v3.csv' 
file = open(data_file_path, 'w', newline='')
csv.writer(file).writerow(['frame', 'pose', 'x', 'y', 'z', 'label'])
file.close()
file = open(data_file_path, 'a', newline='')
writer = csv.writer(file)

# Initialize MediaPipe Pose and Drawing utilities
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

import yt_dlp

@dataclass
class VideoStream:
    url: str = None
    resolution: str = None
    height: int = 0
    width: int = 0

    def __init__(self, video_format):
        self.url = video_format['url']
        self.resolution = video_format['format_note']
        self.height = video_format['height']
        self.width = video_format['width']

    def __str__(self):
        return f'{self.resolution} ({self.height}x{self.width}): {self.url}'

def list_video_streams(url):
    cap = None

    # ℹ️ See help(yt_dlp.YoutubeDL) for a list of available options and public functions
    ydl_opts = {}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

        streams = [VideoStream(format)
                   for format in info['formats'][::-1]
                   if format['vcodec'] != 'none' and 'format_note' in format]
        _, unique_indices = np.unique(np.array([stream.resolution
                                                for stream in streams]), return_index=True)
        streams = [streams[index] for index in np.sort(unique_indices)]
        resolutions = np.array([stream.resolution for stream in streams])
        return streams[::-1], resolutions[::-1]

def cap_from_youtube(url, resolution=None):
    cap = None

    streams, resolutions = list_video_streams(url)
    print(streams)

    if not resolution or resolution == 'best':
        return cv2.VideoCapture(streams[-2].url)

    if resolution not in resolutions:
        raise ValueError(f'Resolution {resolution} not available')
    res_index = np.where(resolutions == resolution)[0][0]
    return cv2.VideoCapture(streams[res_index].url)


#src = input("Enter video url: ")
# Capture frames
#video = cap_from_youtube(src, 'best')
frame_rgb = None
#video = VideoCapture('/Users/rica/Documents/rasp-python-workspace/src/source.webm')
video = VideoCapture('source.webm')
print('START READING')
while video.isOpened():
	# Read frmes from video
	print('START READING')	
	ret, frame = video.read()
	if not ret:
		break
	if frame_rgb is not None:
		cv2.imshow('MediaPipe Pose', frame_rgb)
	frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	
	# Wait a key input
	key = cv2.waitKey(1) & 0xff
	
	# Exit the loop
	if key == 27:
		break
	# Lable frames and write in a csv file
	elif key == ord(' '): 
		print('Wait for labeling')
		label = cv2.waitKey(-1) - 48 # Enter a label
		# Get csv data from a frame
		result = pose.process(frame_rgb)
		mp_drawing.draw_landmarks(frame_rgb, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
		csv_data = write_landmarks_to_csv(result.pose_landmarks.landmark, csv_data, frame_number)
		
		for row in csv_data: # Label frames
			row.append(label)
		writer.writerows(csv_data) # Write a csv file
# Release open resources
file.close()
video.release()
cv2.destroyAllWindows()

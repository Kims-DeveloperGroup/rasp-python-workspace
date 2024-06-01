import cv2
from cv2 import VideoCapture
import matplotlib.pyplot as plt
import numpy as np
import mediapipe as mp
import csv

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

video_src = '~/Users/rica/Documents/rasp-python-workspace/video.h245'
video_src = 0
# Capture frames
frame_rgb = None
video = VideoCapture(video_src)
print('START READING')

# Key guide
# Space: Stop a video for labeling
#  
while video.isOpened():
	# Read frmes from video
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
		label = -1
		while label > 10 or label < 0:
			label = cv2.waitKey(-1) - 48 # Enter a label (0 ~ 9)
		# Get csv data from a frame
		result = pose.process(frame_rgb)
		mp_drawing.draw_landmarks(frame_rgb, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
		csv_data = write_landmarks_to_csv(result.pose_landmarks.landmark)
		
		for row in csv_data: # Label frames
			row.append(label)
		writer.writerows(csv_data) # Write a csv file
# Release open resources
file.close()
video.release()
cv2.destroyAllWindows()

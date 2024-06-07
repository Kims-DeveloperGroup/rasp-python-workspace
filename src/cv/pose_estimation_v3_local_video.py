import cv2
from cv2 import VideoCapture
import matplotlib.pyplot as plt
import numpy as np
import mediapipe as mp
import csv
import time
# 0; Jab, 1: Straigt, 
# 2: Front Hand Hook, 3: Back Hand Hook, 
# 4: Front Hand Upper Cut, 5: Back Hand Upper Cut
# 6: Front Hand Body Shot, 7: Back Hand Body Shot
# 7: Body Jab 8: Body Straigh
labels = ['Jab', 'Straight', 'F-Hook', 'B-Hook', 'F-Upper', 'B-Upper', 'F-Body', 'B-Body', 'Body Jab', 'Body Straight']
import random
def random_label():
	return random.randint(0, 8)

def write_landmarks_to_csv(landmarks, label):
	csv_data = []
	for idx, landmark in enumerate(landmarks):
		print(f"{mp_pose.PoseLandmark(idx).name}: (x: {landmark.x}, y: {landmark.y}, z: {landmark.z})")
		#csv_data.append([ idx, landmark.x, landmark.y, landmark.z])
		csv_data.append(landmark.x)
		csv_data.append(landmark.y)
		csv_data.append(landmark.z)
	print("\n")
	csv_data.append(label)
	return csv_data
timer = 0
def getTime():
	return time.time()
# Init data.csv and open the file writer
data_file_path = '/Users/rica/Documents/data_v3.csv' 
file = open(data_file_path, 'w', newline='')
writer = csv.writer(file)

# Initialize MediaPipe Pose and Drawing utilities
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()
#Text Config
rgb = (255, 0, 0)
font = cv2.FONT_HERSHEY_SIMPLEX
video_src = 0
# Capture frames
frame_rgb = None
video = VideoCapture(video_src)
label = -1
count = 0
started = False
capture_duration = 2 # Time unit:sec
# Key guide
# Space: Stop a video for labeling
while video.isOpened():
	# Read frmes from video
	ret, frame = video.read()
	if not ret:
		break
	if frame_rgb is not None:
		cv2.imshow('MediaPipe Pose', frame_rgb)
	frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	result = pose.process(frame_rgb)
	mp_drawing.draw_landmarks(frame_rgb, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

	# Wait a key input
	key = cv2.waitKey(1) & 0xff
	
	# Exit the loop
	if key == 27:
		break
	elif started == False and key != ord(' '):
		cv2.putText(frame_rgb, 'READY', (50, 500), font, 15.0, rgb, 50)
	elif started == False and key == ord(' '):
		started = True
		cv2.putText(frame_rgb, 'START', (50, 500), font, 15.0, rgb, 50)
	elif count != 0 and count % 10 == 0 : # At dvery 10 dataset, decide to continue or not
		cv2.putText(frame_rgb, 'CONTINUE \nOR NOT', (50, 500), font, 5.0, rgb, 10)
		#key = cv2.waitKey(-1) & 0xff
		if key == 27:# Stop
			break
		else : # Continue
			timer = 0
	# Lable frames and write in a csv file
	elif started == True and label == -1 and timer == 0: 
		label = random_label()
		labelText = labels[label]
		print(f'label={label} value:{labelText}')
		timer = getTime()
	elif label != -1 and timer != 0 and (getTime() - timer) > duration:
		cv2.putText(frame_rgb, "CAPTURE", (50,50), font, 3.0, rgb, 10)
		# Get csv data from a frame
		#result = pose.process(frame_rgb)
		if result.pose_landmarks is not None:
			#mp_drawing.draw_landmarks(frame_rgb, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
			csv_data = write_landmarks_to_csv(result.pose_landmarks.landmark, label)
			#csv_data.append(label)
			writer.writerow(csv_data) # Write a csv file
		label = -1
		timer = 0
		count+=1
	else :
		cv2.putText(frame_rgb, f"{time.time() - timer} sec", (50,50), font, 2.0, rgb, 2)
		cv2.putText(frame_rgb, f"{count}:{labels[label]}", (50, 500), font, 10.0, rgb, 50)
# Release open resources
file.close()
video.release()
cv2.destroyAllWindows()

import imageio
import cv2
from cv2 import VideoCapture
import matplotlib.pyplot as plt
import numpy as np
import mediapipe as mp
import csv

def write_landmarks_to_csv(landmarks, csv_data, frame_number):
    print(f"Landmark coordinates for frame {frame_number}:")
    for idx, landmark in enumerate(landmarks):
        print(f"{mp_pose.PoseLandmark(idx).name}: (x: {landmark.x}, y: {landmark.y}, z: {landmark.z})")
        csv_data.append([frame_number, idx, landmark.x, landmark.y, landmark.z])
    print("\n")

# Init data.csv and open the file writer
data_file_path = '/Users/rica/Documents/data.csv' 
file = open(data_file_path, 'w', newline='')
csv.writer(file).writerow(['frame', 'pose', 'x', 'y', 'z', 'label'])
file.close()
file = open(data_file_path, 'a', newline='')
writer = csv.writer(file)

# Initialize MediaPipe Pose and Drawing utilities
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# Init frame data and video capture
frame_number = -1
csv_data = []

# Capture frames
video = cv2.VideoCapture(0)
started = False
reset = False
frame_rgb = None
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
	elif key == ord(' ') and started == True: 
		print('Wait for labeling')
		label = cv2.waitKey(-1) - 48 # Enter a label
		for row in csv_data: # Label frames
			row.append(label)
		writer.writerows(csv_data) # Write a csv file
		started = False # Stop
	# Begin to read frames
	elif key == ord(' ') and started == False:
		started = True
	# Reset data before begining
	elif started == False and frame_number != 0:
		print('Press space to begin')
		csv_data = []
		frame_number = 0
	elif started == True:
		# Process frames and write frame data
		result = pose.process(frame_rgb)
		if result.pose_landmarks:
			mp_drawing.draw_landmarks(frame_rgb, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
			write_landmarks_to_csv(result.pose_landmarks.landmark, csv_data, frame_number)
		# Increment frame number and convert color to RGB
		frame_number+=1

# Release open resources
file.close()
video.release()
cv2.destroyAllWindows()

import imageio
import cv2
from cv2 import VideoCapture
import matplotlib.pyplot as plt
import numpy as np
import mediapipe as mp
import csv

def write_landmarks_to_csv(landmarks, csv_data, frame_number, writer):
    print(f"Landmark coordinates for frame {frame_number}:")
    for idx, landmark in enumerate(landmarks):
        print(f"{mp_pose.PoseLandmark(idx).name}: (x: {landmark.x}, y: {landmark.y}, z: {landmark.z})")
        writer.writerow([frame_number, idx, landmark.x, landmark.y, landmark.z])
    print("\n")

# init data.csv
data_file_path = '/Users/rica/Documents/data.csv' 
file = open(data_file_path, 'w', newline='')
csv.writer(file).writerow(['frame', 'pose', 'x', 'y', 'z'])
file.close()
file = open(data_file_path, 'a', newline='')
writer = csv.writer(file)
# Initialize MediaPipe Pose and Drawing utilities
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

frame_number = -1
csv_data = []
video = cv2.VideoCapture(0)

# Read frames
while video.isOpened():
	ret, frame = video.read()
	if not ret:
		break
	# Convert the frame to RGB
	frame_number+=1
	frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	
	# Process the frame with MediaPipe Pose
	result = pose.process(frame_rgb)
	# Draw the pose landmarks on the frame
	if result.pose_landmarks:
	    mp_drawing.draw_landmarks(frame_rgb, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
	    write_landmarks_to_csv(result.pose_landmarks.landmark, csv_data, frame_number, writer)
    # Display the frame
	cv2.imshow('MediaPipe Pose', frame_rgb)
	if cv2.waitKey(1) & 0xff ==27:
		break

# Release open resources
file.close()
video.release()
cv2.destroyAllWindows()

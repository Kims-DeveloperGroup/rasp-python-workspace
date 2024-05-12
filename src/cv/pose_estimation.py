import imageio
import cv2
from cv2 import VideoCapture
import matplotlib.pyplot as plt
import numpy as np
import mediapipe as mp

#BaseOptions = mp.tasks.BaseOptions
#PoseLandmarker = mp.tasks.vision.PoseLandmarker
#PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
#VisionRunningMode = mp.tasks.vision.RunningMode

# Create a pose landmarker instance with the video mode:
#options = PoseLandmarkerOptions(
#    base_options=BaseOptions(model_asset_path=model_path),
#    running_mode=VisionRunningMode.VIDEO)

# Initialize MediaPipe Pose and Drawing utilities
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

def write_landmarks_to_csv(landmarks, csv_data, frame_number):
    print(f"Landmark coordinates for frame {frame_number}:")
    for idx, landmark in enumerate(landmarks):
        print(f"{mp_pose.PoseLandmark(idx).name}: (x: {landmark.x}, y: {landmark.y}, z: {landmark.z})")
        csv_data.append([frame_number, mp_pose.PoseLandmark(idx).name, landmark.x, landmark.y, landmark.z])
    print("\n")


frame_number = -1
csv_data = []
video = cv2.VideoCapture(0)
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
	    write_landmarks_to_csv(result.pose_landmarks.landmark, csv_data, frame_number)
    # Display the frame
	cv2.imshow('MediaPipe Pose', frame_rgb)
	if cv2.waitKey(1) & 0xff ==27:
		break
video.release()
cv2.destroyAllWindows()

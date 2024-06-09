import tf.regression_train_csv as t
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
# 8: Body Jab 9: Body Straigh
labels = ['Jab', 'Straight', 'F-Hook', 'B-Hook', 'F-Upper', 'B-Upper', 'F-Body', 'B-Body', 'Body Jab', 'Body Straight']

# Initialize MediaPipe Pose and Drawing utilities
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

import random
def random_label():
	return random.randint(0, 9)

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
def getTime():
	return time.time()
def create_file_writer(path):
	# Init data.csv and open the file writer
	file = open(path, 'w', newline='')
	writer = csv.writer(file)
	column_names = []
	for landmark in mp_pose.PoseLandmark:
		column_names.append(f'{landmark.name}-x')
		column_names.append(f'{landmark.name}-y')
		column_names.append(f'{landmark.name}-z')
	column_names.append('label')
	writer.writerow(column_names)
	return (writer, file)

#
# testRun: if True, model is not trained
# chunk: interval of data size to capture
# labelsToRun: labels of captured data. If the param is empty, labels are randomly generated.
def run(testRun = False, chunk = 10, labelsToRun = [], epochsToTrain = None):
	data_file_path = '/Users/rica/Documents/data_v3.csv' 
	writer, file = create_file_writer(data_file_path)
	
	#Text Config
	rgb = (255, 0, 0)
	font = cv2.FONT_HERSHEY_SIMPLEX
	video_src = 0
	# Capture frames
	frame_rgb = None
	video = VideoCapture(video_src)
	label = -1
	timer = 0
	chunkSize = chunk
	count = 0
	total_data_count = 0
	started = False
	capture_duration = 3 # Time unit:sec
	durationBeforeStart = 70 # frame count
	currDuration = 0
	
	# Store features and labels
	actualLabels = []
	actualFeatures = []
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
		# Give standby time before starting
		elif started == True and currDuration < durationBeforeStart:
			currDuration+=1
			cv2.putText(frame_rgb, f'{(durationBeforeStart - currDuration)/10}', (50, 500), font, 5.0, rgb, 10)
		elif started == False and key != ord(' '):
			cv2.putText(frame_rgb, 'READY', (50, 500), font, 15.0, rgb, 50)
			count = 0
		elif started == False and key == ord(' '):
			started = True
		# At dvery 10 dataset, decide to continue or not
		elif count > 0 and (count % chunkSize) == 0 and started == True : 
			cv2.putText(frame_rgb, 'CONTINUE OR NOT', (50, 500), font, 5.0, rgb, 10)
			if key == ord(' '): # Contiune and ready
				started = False
		# Lable frames and write in a csv file
		elif started == True and label == -1 and timer == 0: 
			if labelsToRun: 
				label = labelsToRun.pop()
			else:
				label = random_label()
			labelText = labels[label]
			print(f'label={label} value:{labelText}')
			timer = getTime()
		elif label != -1 and timer != 0 and (getTime() - timer) > capture_duration:
			cv2.putText(frame_rgb, "CAPTURE", (50,50), font, 3.0, rgb, 10)
			# Get csv data from a frame
			#result = pose.process(frame_rgb)
			if result.pose_landmarks is not None:
				#mp_drawing.draw_landmarks(frame_rgb, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
				csv_data = write_landmarks_to_csv(result.pose_landmarks.landmark, label)
				#csv_data.append(label)
				writer.writerow(csv_data) # Write a csv file
				actualFeatures.append(csv_data.copy().pop())
				actualLabels.append(label)
				total_data_count+=1
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
	model_path = 'tf/models/boxing_pose_est_v1.keras'
	model = None
	if testRun == False:
		epochs = epochsToTrain
		if epochs is None:
			epochs = total_data_count * 20
		t.train(data_file_path, model_path, epochs)
		#model= t.train(data_file_path, model_path, epochs)
	if model is None:
		t.loadFileAndTest(data_file_path, model_path)
	else:
		t.test(features=np.array(actualFeatures), labels=np.array(actualLabels), model=model)

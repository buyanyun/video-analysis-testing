import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
import os, os.path
import time
from scipy import signal


#################################################################
### open_SCIT_json: read all json file in video_name +'_output/' + video_name + '_json/' dir.
### input: video name
### output: video data with the coordination of each body parts (rows is frame, column is body part)
#################################################################

def open_SCIT_json(video_name):

	print("starting extract " + video_name + " JSON files...")

	video_data = []
	size = len([name for name in os.listdir(os.getcwd() + '/' + video_name + '_output/' + video_name + '_json')])

	for i in range(size):
		# input path:
		filename = video_name +'_output/' + video_name + '_json/' + video_name + '_2SCIT_'+ str(i).zfill(12) + '_keypoints.json'

		with open(filename) as json_file:
		    data = json.load(json_file)

		if data['people'] != []:
			dirty_points = np.array(data['people'][0]['pose_keypoints_2d'])

			# zero_idx = -1
			# if (np.count_nonzero(dirty_points) != len(dirty_points)):
			# 	zero_idx = np.where(dirty_points == 0)[0][0] / 3

			n = 25
			x_idx = 3 * np.arange(n)
			y_idx = x_idx + 1

			x_list = dirty_points[x_idx]
			y_list = dirty_points[y_idx]

		else:
			x_list = np.zeros(25)
			y_list = np.zeros(25)

		video_data.append(list(zip(x_list, y_list)))

	return np.array(video_data)


#################################################################
### draw_figure: draw the figure of function
### input: video name, part name, function name, data 
### output: 
#################################################################

def draw_figure(part_name, video_name, function_name, V):
	size = len(V)

	matplotlib.rc('xtick', labelsize=40)
	matplotlib.rc('ytick', labelsize=40)
	plt.figure(figsize=(60,10))
	plt.plot(np.arange(0,size) / 1800, V,'-')
	plt.xticks(np.arange(min(np.arange(0,size)/1800), max(np.arange(0,size)/1800)+1, 1.0))
	plt.xlabel('min', fontsize=40)
	plt.ylabel('pixel/frame', fontsize=40)
	plt.savefig((video_name + '_output/' + part_name +'_' + function_name + '.jpg'))
	plt.clf()

def extract_velocity(part_num, video_data, start_t, end_t):
		V = []
		curr_D = video_data[:,part_num]

		start_f = int(start_t * 1800)
		end_f = int(end_t * 1800)

		# calculate the movement of part in each frame: 5 point stencil
		for i in range(start_f, end_f):

			if curr_D[i+1][0] == 0 or curr_D[i][0] == 0 or curr_D[i+1][1] == 0 or curr_D[i][1] == 0:
				velocity = 0
			else:
				velocity = ((curr_D[i+1][0] - curr_D[i][0])**2 + (curr_D[i+1][1] - curr_D[i][1])**2)**(1/2)
			V.append(velocity)


		# butterworth filter:
		sos = signal.butter(5, 1, 'low',fs=30, output='sos')
		V = signal.sosfilt(sos, V)


		return V

def extract_velocity2(part_num, video_data, start_t, end_t):
		V = []
		curr_D = video_data[:,part_num]

		start_f = int(start_t * 1800)
		end_f = int(end_t * 1800)

		# calculate the movement of part in each frame: 5 point stencil
		for i in range(start_f, end_f):

			if np.sum(curr_D[i:i+5] == 0) != 0:
				velocity = 0
			else:
				fn2h = curr_D[i]
				fnh = curr_D[i+1]
				fh = curr_D[i+2]
				f2h = curr_D[i+3]

				velocity_v = (- f2h + 8*fh - 8*fnh + fn2h) / 12
				velocity = (velocity_v[0]**2 + velocity_v[1]**2)**(1/2)

			V.append(velocity)


		return V

def extract_max_vector_length(part_num, video_data, threshold, start_t, end_t):
		D = []
		curr_D = video_data[:,part_num]

		start_f = int(start_t * 1800)
		end_f = int(end_t * 1800)

		#calculate the movement of part in each frame
		for i in range(start_f, end_f):
			if curr_D[i][0] == 0 or curr_D[i][1] == 0:
				D.append(0)
			else:
				T = []
				for j in range(30):
					if i+j < len(curr_D):
						if curr_D[i+j][0] == 0 or curr_D[i+j][1] == 0:
							T.append(0)
						else:
							velocity = ((curr_D[i+j][0] - curr_D[i][0])**2 + (curr_D[i+j][1] - curr_D[i][1])**2)**(1/2)
							T.append(velocity)
				D.append(np.max(T))

		return D

def num_movement_vector_length(part_num, video_data, start, end, noisy_t, time_t):
	V = extract_max_vector_length(part_num, video_data, time_t, start, end)

	if part_num != 1 and part_num != 8:
		if part_num < 8 or (part_num > 14 and part_num < 19):
			I = extract_max_vector_length(1, video_data, time_t, start, end)
			V = abs(np.subtract(V,I))
		else:
			I = extract_max_vector_length(8, video_data, time_t, start, end)
			V = abs(np.subtract(V,I))

	count = 0
	times = []
	s_times = []
	duration = int(end*1800 - start*1800)
	i = 0

	while (i < duration):
		if V[i] > noisy_t:

			# add count if this frame is the last frame of the movement
			s_times.append(time.strftime("%M:%S", time.gmtime((start*1800 + i) / 30)))

			movement_time = 0
			for j in range(i, duration):
				movement_time = movement_time + 1
				if np.sum(np.array(V[j:j+time_t]) > noisy_t) == 0:
					break
			
			count = count + 1
			i = i + movement_time
			times.append(movement_time)

		else:
			i = i + 1

	return count, times, s_times, V

def num_movement_velocity(part_num, video_data, start, end, noisy_t, time_t):
	V = extract_velocity(part_num, video_data, start, end)

	if part_num != 1 and part_num != 8:
		if part_num < 8 or (part_num > 14 and part_num < 19):
			I = extract_velocity(1, video_data, start, end)
			V = abs(np.subtract(V,I))
		else:
			I = extract_velocity(8, video_data, start, end)
			V = abs(np.subtract(V,I))

	count = 0
	times = []
	s_times = []
	duration = int(end*1800 - start*1800)
	i = 0

	while (i < duration):
		if V[i] > noisy_t:

			# add count if this frame is the last frame of the movement
			s_times.append(time.strftime("%M:%S", time.gmtime((start*1800 + i) / 30)))

			movement_time = 0
			for j in range(i, duration):
				movement_time = movement_time + 1
				if np.sum(np.array(V[j:j+time_t]) > noisy_t) == 0:
					break
			
			count = count + 1
			i = i + movement_time
			times.append(movement_time)

		else:
			i = i + 1

	return count, times, s_times, V

def num_movement_velocity2(part_num, video_data, start, end, noisy_t, time_t):
	V = extract_velocity2(part_num, video_data, start, end)

	if part_num != 1 and part_num != 8:
		if part_num < 8 or (part_num > 14 and part_num < 19):
			I = extract_velocity2(1, video_data, start, end)
			V = abs(np.subtract(V,I))
		else:
			I = extract_velocity2(8, video_data, start, end)
			V = abs(np.subtract(V,I))

	count = 0
	times = []
	s_times = []
	duration = int(end*1800 - start*1800)
	i = 0

	while (i < duration):
		if V[i] > noisy_t:

			# add count if this frame is the last frame of the movement
			s_times.append((start*1800 + i)/30)

			movement_time = 0
			for j in range(i, duration):
				movement_time = movement_time + 1
				if np.sum(np.array(V[j:j+time_t]) > noisy_t) == 0:
					break
			
			count = count + 1
			i = i + movement_time
			times.append(movement_time)

		else:
			i = i + 1

	return count, times, s_times, V







def extract_json_file(video_name, SCIT_start, SCIT_end, time_t): # threshold_v,threshold_v2, threshold_m,:
	video_data = open_SCIT_json(video_name)

	part_names = ["Nose",  "Neck",  "RShoulder",  "RElbow",  "RWrist",  "LShoulder",  \
				"LElbow",  "LWrist",  "MidHip",  "RHip", "RKnee", "RAnkle", "LHip", "LKnee", \
				"LAnkle", "REye", "LEye", "REar", "LEar", "LBigToe", "LSmallToe", "LHeel", \
				"RBigToe", "RSmallToe", "RHeel"]

	# X_v = []
	# X_v2 = []
	X_m = []

	# print(video_name + "Number of movement based on velocity(threshold_v = ", threshold_v, "):")

	# for i in range(len(part_names)):
	# 	V = num_movement_velocity(i, video_data, SCIT_start, SCIT_end, threshold_v, time_t)
	# 	X_v.append(V[0:3])
	# 	print(V[0])
	# 	draw_figure(part_names[i], video_name, 'num_movement_velocity', V[3])

	# print(video_name + "Number of movement based on velocity2(threshold_v = ", threshold_v2, "):")

	# for i in range(len(part_names)):
	# 	V = num_movement_velocity2(i, video_data, SCIT_start, SCIT_end, threshold_v2, time_t)
	# 	X_v2.append(V[0:3])
	# 	print(V[0])
	# 	draw_figure(part_names[i], video_name, 'num_movement_velocity2', V[3])


	# print(video_name + "Number of movement based on magnitude(threshold_m = ", threshold_m, "):")
	print(video_name + "Number of movement based on magnitude:")

	for i in range(len(part_names)):
		if i in [0,1,2,5,8,9,12,15,16,17,18]:
			threshold_m = 1.5
		else:
			threshold_m = 1.5

		V = num_movement_vector_length(i, video_data, SCIT_start, SCIT_end, threshold_m, time_t)
		X_m.append(V[0:3])
		print(V[0])
		draw_figure(part_names[i], video_name, 'num_movement_vector_length', V[3])

	# return X_v, X_v2, X_m
	return X_m



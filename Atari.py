import numpy as ny
import random as rd 
import matplotlib.pyplot as pyplt
import gym
import tensorflow as tsf
from collections import deque

#import time
#import datetime as dt




######################################ENVIRONMENT SETUP##########################################
evs = gym.make('Breakout-v4')
snapshot = evs.reset()
#####################################SETUP DONE##################################################





#########################################GLOBAL VARIABLES######################################
max_batch = 20

discount_factor = 0.99

epo_max = 500

max_time = 2000

img_cropsize = 84

max_mem = 10000

seqno_img = 4
#####################################G_VARIABLES INITIALISED####################################




#graph = tsf.Graph()
seq_of_img = list()




####################################FUNCTION DEFINITIONS#####################################################

def img_op (image):
	sess = tsf.Session()
	with sess.as_default():
		image = image.eval()
		img_press = image.squeeze();
		pyplt.imshow(img_press)
		pyplt.show()


def first_layer_pass (image):
	image = tsf.reshape(image, shape=[-1, 84, 84, 4])

	# Calcuate hidden layer
	hidden_layer_cnn1 = tsf.nn.relu(tsf.nn.conv2d(image, cnn_bnw1, [1, 4, 4, 1], padding = "VALID") + cnn_binary1)
	hidden_layer_cnn2 = tsf.nn.relu(tsf.nn.conv2d(hidden_layer_cnn1, cnn_bnw2, [1, 2, 2, 1], "VALID") + cnn_binary2)
	hidden_layer_cnn2_flat = tsf.reshape(hidden_layer_cnn2,[-1, 2592])
	hidden_final1 = tsf.nn.relu(tsf.matmul(hidden_layer_cnn2_flat, fruncate_bnw1) + fruncate_binary1)

	output = tsf.matmul(hidden_final1,fruncate_bnw2) + fruncate_binary2
	#Quality = tsf.nn.sigmoid(output)
	return output


def Quality_table(val):
	actions = ny.zeros(evs.action_space.n)
	actions[val] = 1
	return actions


def img_processing (image):
	img_color2gray = tsf.image.rgb_to_grayscale(image)
	img_compact = tsf.image.resize_images(img_color2gray, [110, 84])
	env_img = tsf.image.resize_image_with_crop_or_pad(img_compact, img_cropsize, img_cropsize)
	return env_img




###################################END OF FUNCTION DEFINITIONS#############################################################



###################################VARIABLES INITIALISATION########################################################


snapshot_image = tsf.placeholder(tsf.float32, shape=(210, 160, 3))
processed_image = img_processing(snapshot_image)

# Create Quality-Network
image_input = tsf.placeholder(tsf.float32, shape=(None, 4, 84, 84, 1))

cnn_bnw1 = tsf.Variable(tsf.truncated_normal([8,8,4,16], stddev = 0.01))
cnn_binary1 = tsf.Variable(tsf.constant(0.01, shape = [16]))

cnn_bnw2 = tsf.Variable(tsf.truncated_normal([4,4,16,32], stddev = 0.01))
cnn_binary2 = tsf.Variable(tsf.constant(0.01, shape = [32]))

#hidden_layer_cnn2_shape = hidden_layer_cnn2.get_shape().as_list()
#print "dimension:",hidden_layer_cnn2_shape[1]*hidden_layer_cnn2_shape[2]*hidden_layer_cnn2_shape[3]

fruncate_bnw1 = tsf.Variable(tsf.truncated_normal([2592, 256], stddev = 0.01))
fruncate_binary1 = tsf.Variable(tsf.constant(0.01, shape = [256]))

fruncate_bnw2 = tsf.Variable(tsf.truncated_normal([256, evs.action_space.n], stddev = 0.01))
fruncate_binary2 = tsf.Variable(tsf.constant(0.01, shape = [evs.action_space.n]))

#packed_images = tsf.stack([seq_of_img[0], seq_of_img[1], seq_of_img[2], seq_of_img[3]])
Quality = first_layer_pass(image_input)
sq_of_action = tsf.argmax(Quality, 1)

# Training
take_action = tsf.placeholder("float", [None, evs.action_space.n])
yInput = tsf.placeholder("float", [None])
Quality_Action = tsf.reduce_sum(tsf.multiply(Quality, take_action), reduction_indices = 1)
pressure = tsf.reduce_mean(tsf.square(yInput - Quality_Action))
trainStep = tsf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6).minimize(pressure)

memory = list()

operation_initialise = tsf.global_variables_initializer()

savepoint = tsf.train.Saver()



#########################################VARIABLES INITIALISATION COMPLETED#################################################




#########################################START OF WITH BLOCK############################################################
#########################################MAIN PART OF CODE##############################################################

with tsf.Session() as sess:
	sess.run(operation_initialise)

	checkpoint = tsf.train.get_checkpoint_state('./temp/')
        if checkpoint and checkpoint.model_checkpoint_path:
            savepoint.restore(sess, './temp/model.ckpt')
            print('Successfully loaded network weights')
        else:
        	print('Could not find old network weights')

	tym = 0

	for game in range (0, epo_max):
		snapshot = evs.reset()

		total_rewards = 0
		eps=0
		s_eps=1000

		while True:
		    evs.render()
		    eps+=1
		    e=(s_eps-eps)/1000
		    if e<0.1:
		    	e=0.1
		    	
		    console_input = 0

		    next_image = sess.run(processed_image, feed_dict={snapshot_image: snapshot})

		    seq_of_img.append(next_image)

		    # Default console_input
		     
		    if len(seq_of_img) <= seqno_img:
		    	next_snapshot, reward, done, info = evs.step(console_input)

		    else:
		    	seq_of_img.pop(0)
		    	current_state = ny.stack([seq_of_img[0], seq_of_img[1], seq_of_img[2], seq_of_img[3]])#, seq_of_img[4], seq_of_img[5], seq_of_img[6], seq_of_img[7]])

		    	# Get console_input
		    	#e = 0.5
		    	if ny.random.rand(1) < e:
		    		console_input = evs.action_space.sample()
		    	else:
		    		console_input, _Quality = sess.run([sq_of_action, Quality], feed_dict={image_input: [current_state]})

				next_snapshot, reward, done, info = evs.step(console_input) # take a rd console_input

			    # Store in experience relay
				next_image = sess.run(processed_image, feed_dict={snapshot_image: next_snapshot})
				next_state = ny.stack([seq_of_img[1], seq_of_img[2], seq_of_img[3], next_image])# seq_of_img[4], seq_of_img[5], seq_of_img[6], seq_of_img[7], next_image])
				
				#print(Quality_table(console_input))
				action_state = Quality_table(console_input)
				memory.append((current_state, action_state, reward, next_state, done)) 

				if len(memory) > max_mem:
					memory.pop(0)

			# Training
			if tym > max_time:
				# n1 = dt.datetime.now()

				mini = rd.sample(memory, max_batch)

				b_state = [data[0] for data in mini]
				b_action = [data[1] for data in mini]
				b_reward = [data[2] for data in mini]
				b_nextstate = [data[3] for data in mini]
				b_terminal = [data[4] for data in mini]

				# Step 2: calculate y 
				temp_batch = []
				QualityValue_batch = sess.run(Quality, feed_dict={image_input: b_nextstate})

				for i in range(0, max_batch):
					terminal = mini[i][4]
					if terminal:
						temp_batch.append(b_reward[i])
					else:
						temp_batch.append(b_reward[i] + discount_factor * ny.max(QualityValue_batch[i]))

				sess.run(trainStep, feed_dict={yInput: temp_batch, take_action: b_action, image_input: b_state})

				# n2 = dt.datetime.now()
				# print((n2.microsecond - n1.microsecond) / 1e6)

			if tym % 1000 == 0:
				savepoint.save(sess, './temp/model.ckpt')

			total_rewards += reward
		    tym += 1
		    snapshot = next_snapshot


		    if done:
				game += 1
				print('Game Number: {} /n /t Rewards: {}'.format(game, total_rewards))
				break;



############################################END OF WITH BLOCK#############################################################
############################################END OF CODE##################################################################













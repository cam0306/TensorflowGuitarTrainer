""""
Author: Cameron Knight
Description: Uses seq2seq model and tensor flow
to train a guitar learning model
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import inspect

TRAIN = True
path = inspect.os.path.dirname(inspect.os.path.abspath(__file__))
MODEL_SAVE_DIR = path+"/guitar_model.ckpt"
print(MODEL_SAVE_DIR)
AFFERMATIVE = ['y','Y','Yes','YES','yes'] #possible affirmative answers
NEGATIVE = ['n','N','No','NO','no'] #possible negative answers
dataLocation = 'Data/Tabs.txt'
vocab_size = 26 # aka number of frets
n_strings = 6
PAD = [vocab_size - 1] * n_strings
max_seq_length =  50
ZERO_WEIGHT = [0.0] * n_strings
ONE_WEIGHT = [1.0] * n_strings

###DataProcessing####
def GetSong(index):
	"""
	processes a song from the data file at a given
	index in the text file
	returns a list of the notes.
	"""
	with open(dataLocation) as f:
		i = 0
		while(True):
			for line in f:
				if(i == index):
					song = []
					for bar in line.split('|'):
						t_bar = []
						for note in bar.split('~'):
							n = []

							for string in note.split(','):
								if string.strip('\n').isdigit():
									n += [int(string.strip('\n')) + 1]
							if len(n) > n_strings:
								n = n[:n_strings]
							elif len(n) < n_strings:
								n += [0] * (n_strings - len(n))
							if(len(n) != n_strings):
								raise Exception('The data is inconsistant, not enugh stings in data set')
							t_bar += [n]
						t_bar = t_bar[:max_seq_length]
							
						song += [t_bar]
						
					matchingVerses = song[1:] + song[:1]
					return song , matchingVerses
					
				i += 1


###Seq2Seq Network

memory_dim = 250

layers = 12

enc_inp = [tf.placeholder(tf.int32, shape=(n_strings,)) for t in range(max_seq_length)]

labels = [tf.placeholder(tf.int32, shape=(n_strings,)) for t in range(max_seq_length)]

weights = [tf.placeholder(tf.float32,shape=(n_strings,)) for t in range(max_seq_length)]

dec_inp = ([tf.zeros_like(enc_inp[0], dtype=tf.int32, name="GO")]+ enc_inp[:-1])

cell = tf.nn.rnn_cell.LSTMCell(memory_dim)  

cell = tf.nn.rnn_cell.MultiRNNCell([cell] * layers)

dec_outputs,loss =  tf.nn.seq2seq.embedding_rnn_seq2seq(
							enc_inp,
							dec_inp, 
							cell,
							vocab_size,
							vocab_size, 
							vocab_size * 4,
							feed_previous=True
							)

loss = tf.nn.seq2seq.sequence_loss(dec_outputs, labels, weights, vocab_size)

learning_rate = 0.01 
optimizer = tf.train.AdamOptimizer(learning_rate)
train_op = optimizer.minimize(loss)

def train_batch(songIndex,sess):
	"""
	trains the model on a given song index
	returns the average loss of the song
	"""
	t_loss_t = []
	song = GetSong(songIndex)
	if song == None: return
	for bar in song:
		X = bar[0]
		Y = bar[1]

		# builds a dictionary with bucketting of the given song
		feed_dict = {enc_inp[t]: PAD for t in range(max_seq_length)}
		feed_dict.update({labels[t]: PAD for t in range(max_seq_length)})

		feed_dict.update({enc_inp[t]: X[t] for t in range(len(X))})
		feed_dict.update({labels[t]: Y[t] for t in range(len(Y))})
		feed_dict.update({weights[t]: ZERO_WEIGHT for t in range(max_seq_length)})

		feed_dict.update({weights[t]: ONE_WEIGHT for t in range(len(X))})
		feed_dict.update({weights[t]: ONE_WEIGHT for t in range(len(Y))})

		_, loss_t = sess.run([train_op, loss], feed_dict)
		t_loss_t += [float(loss_t)]
	
	return (sum(t_loss_t)/ len (t_loss_t))

saver = tf.train.Saver()

with tf.Session() as sess:

	loss_record = [] # a record of all the losses
	major_loss_record = []# a record of long term average losses

	sess.run(tf.initialize_all_variables()) # initializes the variables
	user_input = ''
	while user_input not in (AFFERMATIVE + NEGATIVE): #checks for a yes or no answer
		# asks if the user would like to train the model or use the model
		user_input = input('Would you like to train the model? (Y/N) ') 
		
	train = (user_input in AFFERMATIVE)
	start_percent = 0
	if(train):
		user_input = ''
		while user_input not in (AFFERMATIVE + NEGATIVE):  #checks for a yes or no answer
			#asks the user if they want to load previose weights
			user_input = input('Would You Like to Train using previouse weights ('+ MODEL_SAVE_DIR+ ')? (Y/N) ') 

		if(user_input in AFFERMATIVE):
			# restores the last used model
			saver.restore(sess, MODEL_SAVE_DIR)
			user_input = ''
			while not user_input.isdigit():
				# has the user select a point in the data to start from
				user_input = input('Enter the starting percentage. 0 to restart form beginning: %') #
			start_percent = .01 * int(user_input)
		else:
			user_input = ''
			while user_input not in (AFFERMATIVE + NEGATIVE):
				# confirms the user wants to overwrite old weights
				user_input = input('Are you Sure you want to overwrite the previouse file ('+ MODEL_SAVE_DIR+ ')? (Y/N) ')
			
			if(user_input in NEGATIVE): 
				exit()
		

		
		training_steps = 500000
		print("Model saved in file:",MODEL_SAVE_DIR)
		for t in range(int(training_steps *(start_percent)),training_steps):
			# Trains on the data
			loss_t = train_batch(t,sess)
			loss_record += [loss_t]
			
			if(t % 500 == 0):
				# saves the model every 500 songs
				saver.save(sess, MODEL_SAVE_DIR)
				average_loss = sum(loss_record)/ len (loss_record)
				print('%{0:.2f} | Loss: {1:.2f}'.format((t / training_steps) * 100, average_loss))
				major_loss_record += [average_loss]
				loss_record = []
		print('Done Training')
	else:
		# restores the model thne runs data through it returning the outputs rather than training the data set
		saver.restore(sess, MODEL_SAVE_DIR)
		print('Data loaded into model.')
		usrInput = 'PlaceHolder'
		while usrInput != '':
			usrInput = input("Enter file name:  ")
			usrTab = []
			lines = []
			with open(usrInput) as f:
				for i in f:
					lines += [i.strip()]
			for i in lines[0]:
				if i.isdigit():
					usrTab += []
				elif i == '-':
					usrTab += []
			for line in lines:
				for char in range(len(usrTab)):
					if line[char].isdigit():
						num = int(line[char]) + 1
						if(char < len(line) -1):
							if(line[char + 1].isdigit()):
								num = num * 10 + int(line[char])

						usrTab[char] += [num]

					elif line[char] == '-':
						usrTab[char] += [0]
					else:
						pass


			feed_dict = {enc_inp[t]: PAD for t in range(max_seq_length)}
			feed_dict.update({enc_inp[t]: usrTab[t] for t in range(len(usrTab))})

			outs= sess.run(dec_outputs, feed_dict)
			processedOut = []
			for i in outs:
				note = []
				for ii in i:
					note += [ii.argmax()]
				processedOut += [note]
			a = (np.array(processedOut).T)
			for i in a:
				print('|',end='')
				for ii in i:
					if ii == 0:
						print('-',end='')
					else:
						print(i-1,end='')
				print('|',end='')
				print()

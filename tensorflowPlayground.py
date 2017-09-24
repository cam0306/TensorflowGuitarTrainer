"""
Author: Cameron Knight
Description: Tensor flow example projects used for reference.
"""


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import tempfile


def create_dataset(hm,variance,step = 2, correlation = False):
	"""
	Genorates a random set of data of lenth hm with the max variance of varance and a pos correlation corespondiong to 
	the slope of step if correlation  == "pos" and negative if correlation = "neg" otherwise no correlation
	"""

	val = 1
	ys = []
	for i in range(hm):
			y =  np.random.normal(val,variance)
			ys.append(y)
			if(correlation and correlation == 'pos'):
				val += step
			if(correlation and correlation == 'neg'):
				val -= step

	xs = [i for i in range(len(ys))]
	return np.array(xs,dtype = np.float32) , np.array(ys,dtype = np.float32)


def LinearRegression(data):
	# print(data[0],data[1])
	d_x = np.array(data[0])
	d_y = np.array(data[1])

	x = tf.placeholder( dtype = tf.float32, shape=(None,))
	y = tf.placeholder( dtype = tf.float32, shape=(None,))
	
	w = tf.Variable(np.random.normal())
	
	p_y = tf.mul(w, x)
	

	cost_op = tf.reduce_mean(tf.square(tf.sub(p_y, y)))
	optim = tf.train.AdamOptimizer(learning_rate=0.1)
	train_op = optim.minimize(cost_op)


	sess = tf.Session()

	sess.run(tf.initialize_all_variables())

	for i in range(200):
		sess.run(train_op, feed_dict={x:d_x,y:d_y})
		cst = sess.run(cost_op, feed_dict={x:d_x,y:d_y})
		print(cst.mean())

	y_pred = sess.run(p_y, feed_dict={x:d_x})
	plt.figure(1)
	plt.scatter(d_x,d_y)
	plt.plot(d_x,y_pred,color='pink')
	plt.show()


def simpleMath():
	tf.reset_default_graph()

	with tf.name_scope('input'):

		a = tf.constant(5.0, name='a')
		b = tf.constant(8.0, name='b')


	with tf.name_scope('output'):

		c = tf.add(a,b,name='c')


	with tf.Session() as sess:
		writer = tf.train.SummaryWriter('/tmp/tensorflow/', graph=tf.get_default_graph())

		sess.run(tf.initialize_all_variables())


		o = sess.run(c)
		print(o)






def QuadraticOprimizer(a,b,c):
	tf.reset_default_graph()

	a = tf.constant(a,dtype=tf.float32)
	b = tf.constant(b,dtype=tf.float32)
	c = tf.constant(c,dtype=tf.float32)

	x = tf.Variable(tf.zeros([1]))

	with tf.name_scope('Calc'):
		y = a*x*x + b*x + c

	with tf.name_scope('train'):
		pos_train_Step = tf.train.AdamOptimizer(0.5).minimize(y)
		neg_train_Step = (tf.train.AdamOptimizer(0.5).minimize(tf.mul(
							tf.constant(-1.0,dtype=tf.float32),y)))

	with tf.Session() as sess:
		writer = tf.train.SummaryWriter('/tmp/tensorflow/', graph=tf.get_default_graph())

		sess.run(tf.initialize_all_variables())
		for i in range(1000):
			if(sess.run(a) > 0):
				sess.run(pos_train_Step)

			else:
				sess.run(neg_train_Step)
		x_,y_ =sess.run([x,y])
		print(x_,y_)



def lstm_Tests():


	np.random.seed(1)      
	size = 1
	batch_size= 100
	n_steps = 45
	seq_width = 50     

	initializer = tf.random_uniform_initializer(-1,1) 

	
	#sequence we will provide at runtime  

	seq_input = tf.placeholder(tf.float32, [n_steps, batch_size, seq_width])
	
	#what timestep we want to stop at

	early_stop = tf.placeholder(tf.int32)

	#inputs = [tf.reshape(i, (batch_size, seq_width)) for i in tf.split(0, n_steps, seq_input)]
	#inputs for rnn needs to be a list, each item being a timestep. 
	#we need to split our input into each timestep, and reshape it because split keeps dims by default  

	#set up lstm

	cell = tf.nn.rnn_cell.LSTMCell(size, initializer=initializer)  
	initial_state = cell.zero_state(batch_size, tf.float32)
	outputs, states = tf.nn.dynamic_rnn(
		cell, 
		seq_input, 
		initial_state=initial_state, 
		time_major = True,
		dtype = tf.float32
		)

	optim = tf.train.AdamOptimizer(learning_rate=0.1)
	train_op = optim.minimize(cost_op)

	#actually initialize, if you don't do this you get errors about uninitialized stuff

	session = tf.Session()
	session.run(tf.initialize_all_variables())

	#define our feeds. 
	#early_stop can be varied, but seq_input needs to match the shape that was defined earlier

	feed = {early_stop:100, seq_input:np.random.rand(n_steps, batch_size, seq_width).astype('float32')}
	
	#run once
	#output is a list, each item being a single timestep. Items at t>early_stop are all 0s

	outs = session.run(outputs, feed_dict=feed)
	

	# for _ in range(30):




	print(type(outs))
	print(len(outs))
	print(outs[-1])

def ShallowLSTM():

	def ConvertToBitArray(num):

		a = "{0:b}".format(num)
		a= ('0' * (16 - len(a))) + a
		d =[int(c) for c in a]
		return d
	def GenPattern(len):
		"""
		returns a numerical patern
		"""
		a = [ConvertToBitArray(i) for i in range(len)]
		b = [ConvertToBitArray(i) for i in range(2*len)]
		return a,b


	sess = tf.Session()
	num_units = 8
	num_proj = 6
	state_size = num_units + num_proj
	batch_size = 3
	input_size = 2

	def encode():
		pass
	x = tf.zeros([batch_size, input_size])
	m = tf.zeros([batch_size, state_size])
	cell = tf.nn.rnn_cell.LSTMCell(
		num_units=num_units, num_proj=num_proj, forget_bias=1.0,
		state_is_tuple=False)
	output, state = cell(x, m)
	sess.run(tf.initialize_all_variables())
	res = sess.run([output, state],
					{x.name: np.array([[1., 1.], [2., 2.], [3., 3.]]),
					m.name: 0.1 * np.ones((batch_size, state_size))})
	print(res)
	#print(GenPattern(5))
	# size = 1
	# x = tf.zeros([1,2])
	# y = tf.zeros([1,2])
	# g, w = tf.nn.rnn_cell.LSTMCell(num_units=num_units, num_proj=num_proj, forget_bias=1.0,
	#	state_is_tuple=False)(x,y)

	# sess = tf.Session()
	# sess.run(tf.initialize_all_variables())

	# a = sess.run(g)
	# b = sess.run(w)
	# print(a)
	# print(b)

	
def seq2seqTest():
	seq_length = 5
	batch_size = 64

	vocab_size = 7
	embedding_dim = 50

	memory_dim = 100


	
	tf.reset_default_graph()
	sess = tf.Session()
	initializer = tf.random_uniform_initializer(-1,1) 
	
	enc_inp = [tf.placeholder(tf.int32, shape=(None,), name="inp%i" % t) for t in range(seq_length)]

	labels = [tf.placeholder(tf.int32, shape=(None,), name="labels%i" % t) for t in range(seq_length)]

	weights = [tf.ones_like(labels_t, dtype=tf.float32) for labels_t in labels]

	# Decoder input: prepend some "GO" token and drop the final
	# token of the encoder input
	dec_inp = ([tf.zeros_like(enc_inp[0], dtype=tf.int32, name="GO")]+ enc_inp[:-1])

	# Initial memory value for recurrence.
	prev_mem = tf.zeros((batch_size, memory_dim))
	
	
	cell = tf.nn.rnn_cell.LSTMCell(memory_dim)  

	dec_outputs, dec_memory = tf.nn.seq2seq.embedding_rnn_seq2seq(
								enc_inp, dec_inp, cell,vocab_size,vocab_size,vocab_size)

								
	loss = tf.nn.seq2seq.sequence_loss(dec_outputs, labels, weights, vocab_size)
	tf.scalar_summary("loss", loss)
	
	magnitude = tf.sqrt(tf.reduce_sum(tf.square(dec_memory[1]))) 
	tf.scalar_summary("magnitude at t=1", magnitude)
	
	summary_op = tf.merge_all_summaries()

	learning_rate = 0.05
	momentum = 0.9
	optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
	train_op = optimizer.minimize(loss)
	
	logdir = tempfile.mkdtemp()
	print(logdir)
	summary_writer = tf.train.SummaryWriter(logdir, sess.graph_def)
	
	sess.run(tf.initialize_all_variables())

	
	def train_batch(batch_size):
		X = [np.random.choice(vocab_size, size=(seq_length,), replace=False)
			 for _ in range(batch_size)]
		Y = X[:]
		
		# Dimshuffle to seq_len * batch_size
		X = np.array(X).T
		Y = np.array(Y).T

		feed_dict = {enc_inp[t]: X[t] for t in range(seq_length)}
		feed_dict.update({labels[t]: Y[t] for t in range(seq_length)})

		_, loss_t, summary = sess.run([train_op, loss, summary_op], feed_dict)
		return loss_t, summary
	
	for t in range(500):
		loss_t, summary = train_batch(batch_size)
		summary_writer.add_summary(summary, t)
	summary_writer.flush()
	
	X_batch = [np.random.choice(vocab_size, size=(seq_length,), replace=False) for _ in range(20)]
	X_batch = np.array(X_batch).T

	feed_dict = {enc_inp[t]: X_batch[t] for t in range(seq_length)}
	dec_outputs_batch = sess.run(dec_outputs, feed_dict)
	
	print(dec_outputs_batch)
	print('\n\n============IN=================')
	[print(list(i)) for i in X_batch]
	print('\n\n============OUT================')
	[print(list(logits_t.argmax(axis=1))) for logits_t in dec_outputs_batch]
	#size = 24
	#outputs, states = tf.nn.seq2seq.embedding_rnn_seq2seq(
	#	encoder_inputs, decoder_inputs, cell,
	#	num_encoder_symbols, num_decoder_symbols,
	#	output_projection=None, feed_previous=False)
#QuadraticOprimizer(2,4,8)
#simpleMath()
#LinearRegression(create_dataset(100000,20000,2.5,correlation='pos'))
#lstm_Tests()
#ShallowLSTM()
seq2seqTest()

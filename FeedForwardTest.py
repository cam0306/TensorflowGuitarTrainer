# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


n_hl1 = 728 //2
n_hl2 = 728
n_hl3 = 728 //2

hm_epochs = 100


# tf Graph Input
x = tf.placeholder("float", shape = (None,784)) #784 is the size of the data input pixels
y = tf.placeholder("float", shape = (None,10))

n_classes = 10 #AKA num output Nodes
batch_size = 100

def NNModel(data):
    #Make Layers    
    hidden1 = {'wieghts':tf.Variable(tf.random_normal([784,n_hl1])), 'biases':tf.Variable(tf.random_normal([n_hl1])) }
    hidden2 = {'wieghts':tf.Variable(tf.random_normal([n_hl1,n_hl2])), 'biases':tf.Variable(tf.random_normal([n_hl2])) }
    hidden3 = {'wieghts':tf.Variable(tf.random_normal([n_hl2,n_hl3])), 'biases':tf.Variable(tf.random_normal([n_hl3])) }
    out_layer = {'wieghts':tf.Variable(tf.random_normal([n_hl3,n_classes])), 'biases':tf.Variable(tf.random_normal([n_classes])) }
    
    #Calculate Laters
    layer1 = tf.add(tf.matmul(data,hidden1['wieghts']),[hidden1['biases']])
    layer1 = tf.nn.elu(layer1)
    
    layer2 = tf.add(tf.matmul(layer1,hidden2['wieghts']),[hidden2['biases']])
    layer2 = tf.nn.elu(layer2)
    
    layer3 = tf.add(tf.matmul(layer2,hidden3['wieghts']),[hidden3['biases']])
    layer3 = tf.nn.elu(layer3)
    
    output = tf.add(tf.matmul(layer3,out_layer['wieghts']),[out_layer['biases']])
    return output
    
def TrainNN(x):
    prediction = NNModel(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction,y))
    
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    acc = [0 for i in range(hm_epochs)]
    epochs = [i for i in range(hm_epochs)]



    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
                
        for epoch in range(hm_epochs):
            epoch_cost = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x,epoch_y = mnist.train.next_batch(batch_size)
                
                _,c = sess.run([optimizer,cost],feed_dict = {x:epoch_x,y:epoch_y})
                epoch_cost += c
            
            print('Epoch:',epoch,"Loss:",epoch_cost)

            correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
            accuracy = tf.reduce_mean(tf.cast(correct,'float'))
            CalcAcc = accuracy.eval({x:mnist.test.images,y:mnist.test.labels})
            acc[epoch] = CalcAcc
            print('Accuracy:',CalcAcc)

    plt.plot(epochs,acc)
    plt.show()
TrainNN(x)

# This is the most basic forward feed neural network to recognize digits

# input data, images of 28 x 28 pixels, grayscale, flatenned to 1 x 784 vectors. https://www.kaggle.com/c/digit-recognizer/data
# This Model
'''
input = N x 784 (each row represents an image)
Number of layers in neural network = 5

Layers
 o  o   o   o   o   o   o   o   o   o   o   o   o           N X 784
 '''''''''''''''''''''''''''''''''''''''''''''''' 
 X  X   X   X   X   X   X   X   X   X   X   X   X           Layer 1: TBD
 '  '  '  '  '  '  '  '  '  '  '  '  '  '  '  '  
    X   X   X   X   X   X   X   X   X   X   X               Layer 2: TBD
      '  '  '  '  '  '  '  '  '  '  '  '  '  
        X   X   X   X   X   X   X   X   X                   Layer 3: TBD
          '  '  '  '  '  '  '  '  '  '
            X   X   X   X   X   X   X                       Layer 4: TBD
              '  '  '  '  '  '  '  '
              X  X  X  X  X  X  X  X                        Layer 5: 10 neurons (one for each digit)
                '  '  '  '  '  '  '                         
                * * * * * * * * * *                         Ouput: vector of size 10 with probabilities from each neuron
 '''

import tensorflow as tf
import numpy as np
import csv
import math
from lib.file_reader import read_chunks


# TODO: see if setting a seed is required at graph level
# It would be needed if we run multiple sessions and want each session to have same random sequence
tf.set_random_seed(0)

input_file = open('data/train.csv',mode='r')
input_reader = csv.reader(input_file, delimiter=',')
next(input_reader) # skip header


total_input_rows = 42000;
mini_batch_size = 50;
# instantiate generator
input_batch=read_chunks(input_reader, mini_batch_size)

# Number of neurons in each layer
L0 = 200
L1 = 100
L2 = 50
L3 = 25
L4 = 10

stddev = 0.1

# setup place holders (inputs that we are going to supply)

# for images
X = tf.placeholder(tf.float32, [None, 784]) # None means we don't know how many images at this point. At run time, it would be batch size

# for correct labels
Y = tf.placeholder(tf.float32, [None, 10])

# setup variables (weights and biases as they need to be predicted by alogithm)

# There are 200 neurons in layer 1. Each neuron requires 784 weights, one for each of the feature/ dimension of the image!
W0 = tf.Variable(tf.truncated_normal([784, L0], stddev = stddev))

# Bias should be positive. Instead of 0, Relu prefers a small positive number.
B0 = tf.Variable(tf.ones(L0)/10) # 0.1

# now we need weights of L1 neurons. Each neuron needs, L0 weights, one for each output of the previous layer
W1 = tf.Variable(tf.truncated_normal([L0, L1], stddev = stddev))
B1 = tf.Variable(tf.ones(L1)/10)

W2 = tf.Variable(tf.truncated_normal([L1, L2], stddev = stddev))
B2 = tf.Variable(tf.ones(L2)/10)

W3 = tf.Variable(tf.truncated_normal([L2, L3], stddev = stddev))
B3 = tf.Variable(tf.ones(L3)/10)

W4 = tf.Variable(tf.truncated_normal([L3, L4], stddev = stddev))
B4 = tf.Variable(tf.ones(L4)/10)


# model setup
Y0 = tf.nn.relu((tf.matmul(X,W0) + B0))
Y1 = tf.nn.relu(tf.matmul(Y0,W1) + B1)
Y2 = tf.nn.relu(tf.matmul(Y1,W2) + B2)
Y3 = tf.nn.relu(tf.matmul(Y2,W3) + B3)
probabilities = tf.matmul(Y3,W4) + B4 
probabilities_scaled = tf.nn.softmax(probabilities)         # softmax is going to scale the probabilities for us!

# cross - entropy:= distance between predictions and truth
# cross_entropy = -tf.reduce_mean(Y_ * tf.log(probabilities_scaled))
# if we use above formula for cross entropy then log(0) can get us into trouble! instead we can  use tf's wrapper

# softmax_cross_entropy uses unscaled probabilities as it would do the scaling for us
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = probabilities, labels = Y)

# way to measure accuracy
correct_predictions = tf.equal(tf.argmax(probabilities_scaled, 1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))


# an optimizer to minimize our distance/ cross entropy
learning_rate = 0.003
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)


init = tf.initialize_all_variables()
session = tf.Session()
session.run(init)

for i in range(1, math.ceil(total_input_rows / mini_batch_size)):
    
    # we need a chunk of data to start processing
    input_chunk = np.array(next(input_batch))   
    
    if input_chunk.size == 0:
        break

    Y_chunk = input_chunk[:,0]
    X_chunk = input_chunk[:,1:]

    Y_chunk_one_hot = (tf.one_hot(indices = Y_chunk, depth = 10)).eval(session=session)

    train_data_dict = { X : X_chunk, Y : Y_chunk_one_hot}

    session.run(optimizer, feed_dict = train_data_dict)

    if (i % 10 == 0):
        print('Iteration: {0}'.format(i))
        a = session.run([accuracy], feed_dict = train_data_dict)
        print("Accuracy: " + str(a)) 

input_file.close()

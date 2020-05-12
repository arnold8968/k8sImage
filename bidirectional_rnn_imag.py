""" Bi-directional Recurrent Neural Network.

A Bi-directional Recurrent Neural Network (LSTM) implementation example using 
TensorFlow library. This example is using the MNIST database of handwritten 
digits (http://yann.lecun.com/exdb/mnist/)

Links:
    [Long Short Term Memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf)
    [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

from __future__ import print_function


import sys

import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import time as time
from sys import stdout
tf.logging.set_verbosity(tf.logging.ERROR)
import contextlib
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import _thread
import psutil

with contextlib.redirect_stdout(None):
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

file_name = sys.argv[1]
pod_name = sys.argv[2]

start_time = time.time()
# root = './'
root = '/root/data'
pid = os.getpid()
p = psutil.Process(pid)

if not os.path.exists(root):
    os.mkdir(root)


import logging
logging.basicConfig(filename=file_name, filemode="w", format="%(message)s",level=logging.DEBUG)
logging.warning("start_time,cur_time,pod_name,num_iteration,total_iteration,loss,gain,epoch_time,cpu")


#log = open(file_name,"a")
cur_time = int(time.time())


tf.set_random_seed(111)

#log.write(str(time.time())+"time,epoch,loss\n")
#log.flush()





# Training Parameters
learning_rate = 0.001
training_steps = 1000
batch_size = 128
display_step = 5

# Network Parameters
num_input = 28 # MNIST data input (img shape: 28*28)
timesteps = 28 # timesteps
num_hidden = 128 # hidden layer num of features
num_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
X = tf.placeholder("float", [None, timesteps, num_input])
Y = tf.placeholder("float", [None, num_classes])

# Define weights
weights = {
    # Hidden layer weights => 2*n_hidden because of forward + backward cells
    'out': tf.Variable(tf.random_normal([2*num_hidden, num_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([num_classes]))
}


def BiRNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, num_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, num_input)
    x = tf.unstack(x, timesteps, 1)

    # Define lstm cells with tensorflow
    # Forward direction cell
    lstm_fw_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
    # Backward direction cell
    lstm_bw_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

    # Get lstm cell output
    try:
        outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                              dtype=tf.float32)
    except Exception: # Old TensorFlow version only returns outputs not states
        outputs = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                        dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

logits = BiRNN(X, weights, biases)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()
saver = tf.train.Saver(max_to_keep=1)


def check_save():
    while True:
        if os.path.exists("/root/"+pod_name+".info") ==True and os.path.exists('/root/{}/{}.meta'.format(pod_name,pod_name)) == False:
        # if os.path.exists('./'+pod_name+'.info')==True and os.path.exists('./{}/{}.meta'.format(pod_name,pod_name))== False:
            
            p = open("/root/"+pod_name+".info","w")
            # p = open("./"+pod_name+".info","w")
            print('get info!')
            p.write(str(epoch))
            p.close()
            # saver.save(sess, './{}/{}'.format(pod_name,pod_name))
            saver.save(sess, '/root/{}/{}'.format(pod_name,pod_name))
            print('save data')
            exit()
            break
        time.sleep(2)

_thread.start_new_thread(check_save,())

last_loss = 100
# Start training
with tf.Session() as sess:


    epoch = 0
    # if os.path.exists("./"+pod_name+".info"):
    if os.path.exists("/root/"+pod_name+".info"):
        while True:
            print('restore')
            # if os.path.exists('./{}/{}.meta'.format(pod_name,pod_name)):

            if os.path.exists('/root/{}/{}.meta'.format(pod_name,pod_name)):
                sess.run(init)
                saver = tf.train.Saver()
                saver.restore(sess,'/root/{}/{}'.format(pod_name,pod_name))
                # saver.restore(sess,'./{}/{}'.format(pod_name,pod_name))
                try:
                    info_f = open("/root/"+pod_name+".info","r")
                    # info_f = open("./"+pod_name+".info","r")
                    init_epoch = int(info_f.read())
                    info_f.close()
                    print(init_epoch >1)
                except Exception as e:
                    print(e)
                    continue
            break
    else:
        sess.run(init)
        init_epoch = 1
        print("load model," + str(int(time.time()) - cur_time) + ",\n")


    for epoch in range(init_epoch, training_steps+1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Reshape data to get 28 seq of 28 elements
        batch_x = batch_x.reshape((batch_size, timesteps, num_input))
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
            # Calculate batch loss and accuracy
        loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})
        print(" Epoch: {}  Time: {:.6f}  Loss: {:.5f}".format(epoch,time.time(), loss))
        stdout.flush()
        logging.warning("{},{},{},{},{},{},{},{},{}".format(str(start_time),str(time.time()),pod_name,str(epoch),\
                                               str(training_steps),loss,str(last_loss-loss),str(time.time()-cur_time),str(p.cpu_percent())))
        cur_time = time.time()

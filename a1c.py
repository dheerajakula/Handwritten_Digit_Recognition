""" Starter code for simple logistic regression model for MNIST
with tf.data module
MNIST dataset: yann.lecun.com/exdb/mnist/
Created by Chip Huyen (chiphuyen@cs.stanford.edu)
CS20: "TensorFlow for Deep Learning Research"
cs20.stanford.edu
Lecture 03
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import tensorflow as tf
import time

import utils

# Define paramaters for the model
learning_rate = 0.005
batch_size = 64
n_epochs = 32
n_train = 60000
n_test = 10000

# Step 1: Read in data
data = 'data'
mnist_folder = 'data/mnist'
if os.path.isdir(mnist_folder) != True:
    os.mkdir(data)
    os.mkdir(mnist_folder)
utils.download_mnist(mnist_folder)
train, val, test = utils.read_mnist(mnist_folder, flatten=True)

# Step 2: Create datasets and iterator
# create training Dataset and batch it
train_data = tf.data.Dataset.from_tensor_slices(train)
train_data = train_data.shuffle(10000) # if you want to shuffle your data
train_data = train_data.batch(batch_size)

# create testing Dataset and batch it
test_data = tf.data.Dataset.from_tensor_slices(test)
test_data = test_data.batch(batch_size)

#############################
########## TO DO ############
#############################


# create one iterator and initialize it with different datasets
iterator = tf.data.Iterator.from_structure(train_data.output_types, 
                                           train_data.output_shapes)
img, label = iterator.get_next()

train_init = iterator.make_initializer(train_data)	# initializer for train_data
test_init = iterator.make_initializer(test_data)	# initializer for train_data
keep_prob = tf.placeholder(tf.float32)

# Step 3: create weights and bias
# w is initialized to random variables with mean of 0, stddev of 0.01
# b is initialized to 0
# shape of w depends on the dimension of X and Y so that Y = tf.matmul(X, w)
# shape of b depends on Y
w, b = tf.get_variable(name='weights', shape=(784, 10), initializer=tf.random_normal_initializer(0, 0.01)), tf.get_variable(name='bias', shape=(1, 10), initializer=tf.zeros_initializer())
#############################
########## TO DO ############
#############################
n_hidden_1 = 1024 # 1st layer number of neurons
n_hidden_2 = 256 # 2nd layer number of neurons
num_input = 784 # MNIST data input (img shape: 28*28)
num_classes = 10 # MNIST total classes (0-9 digits)

weights = {
    'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, num_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}

# Step 4: build model
# the model that returns the logits.
# this logits will be later passed through softmax layer
def neural_net(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

#logits = neural_net(img)
logits = neural_net(img)
#############################
########## TO DO ############
#############################


# Step 5: define loss function
# use cross entropy of softmax of logits as the loss function
entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label, name='entropy')
loss = tf.reduce_mean(entropy, name='loss') 
#############################
########## TO DO ############
#############################


# Step 6: define optimizer
# using Adamn Optimizer with pre-defined learning rate to minimize loss
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
#############################
########## TO DO ############
#############################


# Step 7: calculate accuracy with test set
preds = tf.nn.softmax(logits)
correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(label, 1))
accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

writer = tf.summary.FileWriter('./graphs/dnn', tf.get_default_graph())

with tf.Session() as sess:
   
    start_time = time.time()
    sess.run(tf.global_variables_initializer())

    # Create a summary to monitor cost tensor
    tf.summary.scalar("loss", loss)
    # Create a summary to monitor accuracy tensor
    tf.summary.scalar("accuracy", accuracy)
    # Merge all summaries into a single op
    merged_summary_op = tf.summary.merge_all()

    # train the model n_epochs times
    itern = 0
    for i in range(n_epochs): 	
        sess.run(train_init)	# drawing samples from train_data
        total_loss = 0
        n_batches = 0

        try:
            while True:
                _, l, summary= sess.run([optimizer, loss, merged_summary_op], feed_dict={keep_prob: 0.9})

                total_loss += l
                n_batches += 1

                #writer.add_summary(summary, itern )
                itern += 1

                
        except tf.errors.OutOfRangeError:
            pass
        print('Training Average loss epoch {0}: {1}'.format(i, total_loss/n_batches))
        
        sess.run(test_init)
        total_correct_preds = 0
        try:
            while True:
                accuracy_batch = sess.run(accuracy, feed_dict={keep_prob: 0.9})
                total_correct_preds += accuracy_batch
        except tf.errors.OutOfRangeError:
            pass

        print('Testing Accuracy epoch {0}: {1}%'.format(i, (total_correct_preds/n_test)*100))
    print('Total time: {0} seconds'.format(time.time() - start_time))

    # test the model
    sess.run(test_init)			# drawing samples from test_data
    total_correct_preds = 0
    try:
        while True:
            accuracy_batch = sess.run(accuracy, feed_dict={keep_prob: 1.0})
            total_correct_preds += accuracy_batch
    except tf.errors.OutOfRangeError:
        pass

    print('Final Testing Accuracy after {0} epochs is {1}% '.format(n_epochs, (total_correct_preds/n_test)*100))
writer.close()

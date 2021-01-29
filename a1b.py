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
batch_size = 128
n_epochs = 10
n_train = 60000
n_test = 10000
num_classes = 10

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
#############################
########## TO DO ############
#############################

# Store layers weight & bias
weights_n_biases = {
    # 3x3 conv, 1 input, 16 outputs
    'conv1_weights': tf.Variable(tf.random_normal([3, 3, 1, 16])),
    # 3x3 conv, 16 inputs, 32 outputs
    'conv2_weights': tf.Variable(tf.random_normal([3, 3, 16, 32])),
    # 3x3 conv, 32 inputs, 64 outputs
    'conv3_weights': tf.Variable(tf.random_normal([3, 3, 32, 64])),
    # fully connected, 6*6*64 inputs, 1024 outputs
    'hiddenlayer_weights': tf.Variable(tf.random_normal([6*6*64, 512])),
    # 512 inputs, 10 outputs (class prediction)
    'outputlayer_weights': tf.Variable(tf.random_normal([512, num_classes])),
    # biases for 16 layers output from conv1
    'conv1_biases': tf.Variable(tf.random_normal([16])),
    # biases for 32 layers output from conv2
    'conv2_biases': tf.Variable(tf.random_normal([32])),
    # biases for 64 layers output from conv3
    'conv3_biases': tf.Variable(tf.random_normal([64])),
    # biases for 64 outputs from neural layer
    'hiddenlayer_biases': tf.Variable(tf.random_normal([512])),
    # biases for 10 outputs from final layer
    'outputlayer_biases': tf.Variable(tf.random_normal([10]))
}

def conv_net(x, weights_n_biases, dropout):
    
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # Convolution Layer
    conv1 = tf.nn.conv2d(x, weights_n_biases['conv1_weights'], strides = [1,1,1,1], padding = 'SAME' )
    conv1 = tf.nn.bias_add(conv1, weights_n_biases['conv1_biases'])
    # Max Pooling (down-sampling)
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='SAME')

    # Convolution Layer
    conv2 = tf.nn.conv2d(conv1, weights_n_biases['conv2_weights'], strides = [1,1,1,1], padding = 'SAME' )
    conv2 = tf.nn.bias_add(conv2, weights_n_biases['conv2_biases'])
    # Max Pooling (down-sampling)
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='SAME')

   
    # Convolution Layer
    conv3 = tf.nn.conv2d(conv2, weights_n_biases['conv3_weights'], strides = [1,1,1,1], padding = 'SAME' )
    conv3 = tf.nn.bias_add(conv3, weights_n_biases['conv3_biases'])
    # Max Pooling (down-sampling)
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='VALID')

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    hiddenlayer = tf.reshape(conv3, [-1, weights_n_biases['hiddenlayer_weights'].get_shape().as_list()[0]])
    hiddenlayer = tf.add(tf.matmul(hiddenlayer, weights_n_biases['hiddenlayer_weights']), weights_n_biases['hiddenlayer_biases'])
    hiddenlayer = tf.nn.relu(hiddenlayer)
    # Apply Dropout
    hiddenlayer = tf.nn.dropout(hiddenlayer, dropout)

    # Output, class prediction
    outputlayer = tf.add(tf.matmul(hiddenlayer, weights_n_biases['outputlayer_weights']), weights_n_biases['outputlayer_biases'])
    return outputlayer

logits = conv_net(img, weights_n_biases, keep_prob)
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

writer = tf.summary.FileWriter('./graphs/cnn', tf.get_default_graph())

with tf.Session() as sess:
   
    start_time = time.time()
    sess.run(tf.global_variables_initializer())

    t_l_1 = tf.summary.scalar("training loss", loss)

    t_l_2 = tf.summary.scalar("testing loss", loss)
    
    itern_train = 0
    itern_test = 0
    # train the model n_epochs times
    for i in range(n_epochs): 	
        sess.run(train_init)	# drawing samples from train_data
        total_loss = 0
        n_batches = 0
        try:
            while True:
                _, l, summary= sess.run([optimizer, loss, t_l_1], feed_dict={keep_prob: 0.9})
                total_loss += l
                n_batches += 1
                itern_train += 1
                writer.add_summary(summary, itern_train )

                
        except tf.errors.OutOfRangeError:
            pass
        print('Training Average loss epoch {0}: {1}'.format(i, total_loss/n_batches))

        n_batches = 0
        sess.run(test_init)
        total_correct_preds = 0
        try:
            while True:
                accuracy_batch, l, summary= sess.run([accuracy,loss, t_l_2], feed_dict={keep_prob: 1.0})
                total_loss += l
                n_batches += 1
                itern_test += 1
                total_correct_preds += accuracy_batch
                writer.add_summary(summary, itern_test )


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
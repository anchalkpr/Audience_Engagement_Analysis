
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

from skimage.feature import hog
from skimage import data, color, exposure
from PIL import Image
import matplotlib.pyplot as plt

import os
from random import shuffle

from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn import metrics

import tensorflow as tf


# In[2]:


data_path = 'wacv2016/dataset'

person_list = ['11', '20', '23', '18', '2', '12', '13',
                 '19', '14', '22', '3', '8', '6', '17',
                 '4', '10', '5', '1', '16', '9', '7', '21']

classes = os.listdir(data_path)
classes.remove('.DS_Store')

person_sets = set()

for category in classes:
    img_path = os.path.join(data_path, category)
    image_files = os.listdir(img_path)
    for img_file in image_files:
        person_id = img_file.split('_')[0][3:]
        person_sets.add(person_id)
        
dev_persons = ['1', '20', '14', '7', '9', '4',
               '6', '8', '3', '10', '5', '11',
               '18', '16', '12', '22', '23', '19']
test_persons = ['2', '13', '17', '21']

shuffle(dev_persons)
train_persons = dev_persons[:-2]
val_persons = dev_persons[-2:]


# In[3]:


xtrain = list()
xval = list()
xtest = list()
ytrain = list()
yval = list()
ytest = list()

xdev = list()
ydev = list()

for category in classes:
    img_path = os.path.join(data_path, category)
    image_files = os.listdir(img_path)
    if '.DS_Store' in image_files:
        image_files.remove('.DS_Store')
        print(len(image_files))
        
    for img_file in image_files:
        person_id = img_file.split('_')[0][3:]
        
        img = np.array(Image.open(os.path.join(img_path,img_file)))
        if img.shape != (100,100):
            continue
        
        img = (img.flatten()/255.0)
        if person_id in train_persons:
            xtrain.append(img)
            xdev.append(img)
            ytrain.append(int(category))
            ydev.append(int(category))
        elif person_id in val_persons:
            xval.append(img)
            xdev.append(img)
            yval.append(int(category))
            ydev.append(int(category))
        elif person_id in test_persons:
            xtest.append(img)
            ytest.append(int(category))
            
xtrain = np.array(xtrain)
ytrain = np.array(ytrain)

# shuffle training data
s = np.arange(xtrain.shape[0])
np.random.shuffle(s)
print(s)

xtrain = xtrain[s]
ytrain = ytrain[s]

xval = np.array(xval)
yval = np.array(yval)
xtest = np.array(xtest)
ytest = np.array(ytest)

xdev = np.array(xdev)
ydev = np.array(ydev)


# In[4]:


cov_ytrain = np.zeros((ytrain.shape[0], 3))
for i in range(ytrain.shape[0]):
    cov_ytrain[i][ytrain[i]-1] = 1
    
cov_yval = np.zeros((yval.shape[0], 3))
for i in range(yval.shape[0]):
    cov_yval[i][yval[i]-1] = 1
    
cov_ytest = np.zeros((ytest.shape[0], 3))
for i in range(ytest.shape[0]):
    cov_ytest[i][ytest[i]-1] = 1
    
class_weights = [xtrain.shape[0]/np.sum(yval == i) for i in range(1,4)]
class_weights = ((class_weights-np.amin(class_weights))*4)/(np.amax(class_weights)-np.amin(class_weights)) + 1


# # Convolutional network

# In[5]:


# Functions for convolutional networks
def weight_variable(shape, name):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial, name=name)

def bias_variable(shape, name):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial, name=name)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME') # strides=[batch, x, y, channel]

def max_pool_2x2(x, padding='SAME'):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding=padding)


# In[6]:


x = tf.placeholder(tf.float32, shape=[None, 100, 100, 1])
y_ = tf.placeholder(tf.float32, shape=[None, 3])
class_w = tf.constant(class_weights, dtype=tf.float32)


# In[7]:


# First layer
W_conv1 = weight_variable([5, 5, 1, 96], name='conv1_w')
b_conv1 = bias_variable([96], name='conv1_b')

h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)

# Second layer
W_conv2 = weight_variable([5, 5, 96, 64], name='conv2_w')
b_conv2 = bias_variable([64], name='conv2_b')

h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)

h_pool2 = max_pool_2x2(h_conv2)


# First densely connected layer
W_fc1 = weight_variable([50 * 50 * 64, 96], name='fc1_w')
b_fc1 = bias_variable([96], name='fc1_b')

h_pool2_flat = tf.reshape(h_pool2, [-1, 50*50*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
'''
# Second Densely connected layer
W_fc2 = weight_variable([512, 256], name='fc2_w')
b_fc2 = bias_variable([256], name='fc2_b')

h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)
'''
# Readout layer
W_fc3 = weight_variable([96, 3], name='fc3_w')
b_fc3 = bias_variable([3], name='fc3_b')

y_conv = tf.matmul(h_fc1_drop, W_fc3) + b_fc3


# In[8]:


# Optimization
sample_weights = tf.reduce_sum(class_w * y_conv, axis=1)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)
weighted_loss = tf.reduce_mean(cross_entropy*sample_weights)
train_step = tf.train.AdamOptimizer(0.1).minimize(weighted_loss)

prediction = tf.argmax(y_conv, 1)

# Calculating the error on test data
mistakes = tf.not_equal(tf.argmax(y_, 1), tf.argmax(y_conv, 1))
error = tf.reduce_mean(tf.cast(mistakes, tf.float32))


# In[10]:


batch_size = 200
instance_num = xtrain.shape[0]
num_batch = int(np.ceil(float(instance_num)/batch_size))

session = tf.InteractiveSession()
tf.global_variables_initializer().run()
saver = tf.train.Saver()

for i in range(10):
    for j in range(num_batch):
        if j < num_batch-1:
            train_input = xtrain[j*batch_size:(j*batch_size) + batch_size]
            train_labels = cov_ytrain[j*batch_size:(j*batch_size) + batch_size]
        else:
            train_input = xtrain[j*batch_size:]
            train_labels = cov_ytrain[j*batch_size:]
        
        train_step.run(feed_dict={x: train_input.reshape((train_input.shape[0],100,100,1)), y_: train_labels,
                                  class_w: class_weights, keep_prob: 0.5})
        
        if i % 2 == 0:
            error_rate = error.eval(feed_dict={
                        x: train_input.reshape((train_input.shape[0],100,100,1)), y_: train_labels, keep_prob: 1.0})
            print('Epoch {}, batch {}: error rate = {}'.format(i, j, error_rate))
            

val_error_rate = error.eval(feed_dict={x: xval.reshape((xval.shape[0],100,100,1)), y_: yval, keep_prob: 1.0})
print('Validation error = {}'.format(val_error_rate))

save_path = saver.save(session, "convolutional.ckpt")
print("Model saved in file: %s" % save_path)


#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function
import matplotlib.pyplot as plt

import tensorflow as tf
#from tensorflow.python.tools import inspect_checkpoint as chkp
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes

import numpy as np
import os
from os import listdir
from os.path import isfile, join, isdir
from random import shuffle, choice
from PIL import Image
import sys
import json
import collections, cv2


input_width = 224
input_height = 224
num_channels = 3
slim = tf.contrib.slim
n_hidden1 = 4096
n_hidden2 = 4096
feature_size = 4096
learnError = 0
n_epochs = 1
batch_size = 2
min_steps = batch_size

lr = 1e-10


def loadData(jsonData, inPath):
    batchPaths = []
    for vid in jsonData.keys():
        # VIRAT format
        # dirName = '_'.join(vid.split('.')[0].split('_')[2:])

        # Other dataset file name format
        dirName = ''.join(vid.split('.')[0])
        # print(dirName)
        vidPath = join(inPath,dirName)
        #print(vidPath)
        #batchPaths = batchPaths + sorted([str(join(vidPath, f) + '/') for f in listdir(vidPath) if isdir(join(vidPath, f))])

        # Breakfast Actions
        batchPaths = batchPaths + [vidPath]
        #print(batchPaths)
    return batchPaths

def loadMiniBatch(vidFilePath):
    #vidName = vidFilePath.split('/')[-3]
    frameList = sorted([join(vidFilePath, f) for f in listdir(vidFilePath) if isfile(join(vidFilePath, f)) and f.endswith('.jpg')])
    frameList = sorted(frameList, key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[1]))
    its = [iter(frameList), iter(frameList[1:])]
    segments = zip(*its)
    minibatch = []
    for segment in segments:
        # print(segment)
        im = []
        numFrames = 0
        for j, imFile in enumerate(segment):
            img = Image.open(imFile)
            img = img.resize((input_width, input_height), Image.ANTIALIAS)
            img = np.array(img)
            img = img[:, :, ::-1].copy()
#            img = cv2.GaussianBlur(img,(5,5),0)
            im.append(img)
            numFrames += 1
        minibatch.append(np.stack(im))
    return vidFilePath, minibatch

def lstm_cell(W, b, forget_bias, inputs, state):
    one = constant_op.constant(1, dtype=dtypes.int32)
    add = math_ops.add
    multiply = math_ops.multiply
    sigmoid = math_ops.sigmoid
    activation = math_ops.sigmoid
    # activation = math_ops.tanh

    c, h = array_ops.split(value=state, num_or_size_splits=2, axis=one)

    gate_inputs = math_ops.matmul(array_ops.concat([inputs, h], 1), W)
    gate_inputs = nn_ops.bias_add(gate_inputs, b)
    # i = input_gate, j = new_input, f = forget_gate, o = output_gate
    i, j, f, o = array_ops.split(value=gate_inputs, num_or_size_splits=4, axis=one)

    forget_bias_tensor = constant_op.constant(forget_bias, dtype=f.dtype)

    new_c = add(multiply(c, sigmoid(add(f, forget_bias_tensor))), multiply(sigmoid(i), activation(j)))
    new_h = multiply(activation(new_c), sigmoid(o))
    new_state = array_ops.concat([new_c, new_h], 1)

    return new_h, new_state

# The following vgg_16 adapted from
# https://github.com/tensorflow/models/blob/master/research/slim/nets/vgg.py
def vgg_16(inputs,
           is_training=True,
           dropout_keep_prob=0.5,
           scope='vgg_16',
           fc_conv_padding='VALID'):
  """Oxford Net VGG 16-Layers version D Example.
  Note: All the fully_connected layers have been transformed to conv2d layers.
        To use in classification mode, resize input to 224x224.
  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes. If 0 or None, the logits layer is
      omitted and the input features to the logits layer are returned instead.
    is_training: whether or not the model is being trained.
    dropout_keep_prob: the probability that activations are kept in the dropout
      layers during training.
    spatial_squeeze: whether or not should squeeze the spatial dimensions of the
      outputs. Useful to remove unnecessary dimensions for classification.
    scope: Optional scope for the variables.
    fc_conv_padding: the type of padding to use for the fully connected layer
      that is implemented as a convolutional layer. Use 'SAME' padding if you
      are applying the network in a fully convolutional manner and want to
      get a prediction map downsampled by a factor of 32 as an output.
      Otherwise, the output prediction map will be (input / 32) - 6 in case of
      'VALID' padding.
    global_pool: Optional boolean flag. If True, the input to the classification
      layer is avgpooled to size 1x1, for any input size. (This is not part
      of the original VGG architecture.)
  Returns:
    net: the output of the logits layer (if num_classes is a non-zero integer),
      or the input to the logits layer (if num_classes is 0 or None).
    end_points: a dict of tensors with intermediate activations.
  """
  with tf.variable_scope(scope, 'vgg_16', [inputs]) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                        outputs_collections=end_points_collection):
      net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
      net = slim.max_pool2d(net, [2, 2], scope='pool1')
      net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
      net = slim.max_pool2d(net, [2, 2], scope='pool2')
      net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
      net = slim.max_pool2d(net, [2, 2], scope='pool3')
      net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
      net = slim.max_pool2d(net, [2, 2], scope='pool4')
      net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
      net = slim.max_pool2d(net, [2, 2], scope='pool5')

      # Use conv2d instead of fully_connected layers.
      net = slim.conv2d(net, 4096, [7, 7], padding=fc_conv_padding, scope='fc6')
      net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                         scope='dropout6')
      net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
      vgg16_features = tf.reshape(net, (-1,4096))
      # Convert end_points_collection into a end_point dict.
      end_points = slim.utils.convert_collection_to_dict(end_points_collection)
return net, end_points, vgg16_features

#----------------------------------------------------------------
#    MAIN CODE SECTION
#jsonData = json.load(open(sys.argv[1]))
#vidPath = sys.argv[2]
#modelPath = sys.argv[3]
#activeLearningInput = sys.argv[4]

#python Zacks_VGG_RNN.py <jsonData path> <video frames root directory> <path to restore model> <output file name to write loss characteristics> ```

jsonData = json.load(open('frameNames.json'))
vidPath = 'newCaledonia'
modelPath = 'model'
activeLearningInput = 1

if activeLearningInput == "1":
    activeLearning = True
else:
    activeLearning = False

# creates a list of all the file names by concatenating the directory names
# in vidPath and jsonData and the files therein
batch = loadData(jsonData, vidPath)
print(batch)
tf.compat.v1.reset_default_graph()
# ----------------------------------------------------- #
# declaring the variables that will be needed
inputs = tf.compat.v1.placeholder(tf.float32, (None, 224, 224, 3), name='inputs')
learning_rate = tf.compat.v1.placeholder(tf.float32, [])
is_training = tf.compat.v1.placeholder(tf.bool)

# Setup LSTM
init_state1 = tf.compat.v1.placeholder(tf.float32, [1, 2*n_hidden1], name="State")
W_lstm1 = vs.get_variable("W1", shape=[feature_size + n_hidden1, 4*n_hidden1])
b_lstm1 = vs.get_variable("b1", shape=[4*n_hidden1], initializer=init_ops.zeros_initializer(dtype=tf.float32))
curr_state1 = init_state1

# ----------------------------------------------------- #
#             SETTING UP VGG


r, g, b = tf.split(axis=3, num_or_size_splits=3, value=inputs * 255.0)
VGG_MEAN = [103.939, 116.779, 123.68]
VGG_inputs = tf.concat(values=[b - VGG_MEAN[0], g - VGG_MEAN[1], r - VGG_MEAN[2]], axis=3)

#tf.summary.image(name='Input Image', tensor= VGG_inputs)

net, end_points, vgg16_Features = vgg_16(inputs=VGG_inputs, is_training=True,
                                         dropout_keep_prob=0.8,
                                         scope='vgg_16', fc_conv_padding='VALID')

RNN_inputs = tf.reshape(vgg16_Features[0,:], (-1, feature_size))
#tf.summary.image(name='VGG output', tensor= tf.reshape(RNN_inputs, (-1, 64, 64, 1)))

# LSTM
h_1, curr_state1 = lstm_cell(W_lstm1, b_lstm1, 1.0, RNN_inputs, curr_state1)
fc1 = h_1
#tf.summary.image(name='LSTM output', tensor= tf.reshape(fc1, (-1, 64, 64, 1)))

print(fc1.shape, vgg16_Features.shape)
sseLoss1 = tf.square(tf.subtract(fc1[0,:], vgg16_Features[1,:]))
mask = tf.greater(sseLoss1, learnError * tf.ones_like(sseLoss1))
sseLoss1 = tf.multiply(sseLoss1, tf.cast(mask, tf.float32))
sseLoss = tf.reduce_mean(sseLoss1)
tf.summary.scalar(name='SSE loss', tensor=sseLoss)
# Optimization
train_op = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(sseLoss)

#####################
### Training loop ###
#####################

print(tf.global_variables())
init = tf.compat.v1.global_variables_initializer()

merged = tf.summary.merge_all()

saver = tf.compat.v1.train.Saver(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope="vgg_16"))
with tf.compat.v1.Session() as sess:
    # Initialize parameters
    sess.run(init)
    saver.restore(sess, "./vgg_16.ckpt")
    saver = tf.compat.v1.train.Saver(max_to_keep=0)
    # Merge all the summaries and write them
    file_writer = tf.summary.FileWriter('./log', sess.graph)
    avgPredError = 1.0
    # LSTM
    new_state = np.random.uniform(-0.5,high=0.5,size=(1,2*n_hidden1))
    for i in range(n_epochs):
        # Run 1 epoch
        loss = []
		#shuffle the sequences
        shuffle(batch)
        for miniBatchPath in batch:
            # LSTM
            new_state = np.random.uniform(-0.5,high=0.5,size=(1,2*n_hidden1))
            avgPredError = 0
            vidName, minibatches = loadMiniBatch(miniBatchPath)
            segCount = 0
            predError = collections.deque(maxlen=30)
            print('Video:', vidName)
            for x_train in minibatches:
                segCount += 1
                ret = sess.run([train_op, sseLoss, sseLoss1, curr_state1, fc1, merged],
				                feed_dict = {inputs: x_train, is_training: True, init_state1: new_state, learning_rate:lr})
                new_state = ret[3]
                #print ('ret =', ret)
                # add the summary to the writer (i.e. to the event file)
                file_writer.add_summary(ret[5], segCount)

                if activeLearning:
                    if ret[1]/avgPredError > 1.5:
                        lr = 1e-8
                        print('Gating n_steps=', segCount, avgPredError, ret[1])
                        # predError.clear()
                    else:
                        # print('NOT Gating n_steps=', segCount, avgPredError, ret[1])
                         lr = 1e-10
                predError.append(ret[1])
                avgPredError = np.mean(predError)

        path = modelPath + str(i+1)
        save_path = saver.save(sess, path)

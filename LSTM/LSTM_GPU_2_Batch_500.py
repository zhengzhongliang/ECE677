from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data
from six.moves import xrange
import tensorflow as tf
import numpy as np
import time
from tensorflow.contrib import rnn


FLAGS = None

N_EPOCH=40
BATCH_SIZE=500
VOLUME = 50000
N_GPU=2

N_BATCH=int(VOLUME/BATCH_SIZE/N_GPU)
TOWER_SIZE=BATCH_SIZE*N_BATCH

MOVING_AVERAGE_DECAY=0.999
SHOW_PLACEMENT=False
LR=0.01
SEED=2

N_INPUT_DIM = 28 # MNIST data input (img shape: 28*28)
N_STEP = 28 # timesteps
N_HIDDEN = 128 # hidden layer num of features
N_CLASS = 10 # MNIST total classes (0-9 digits)



def MLP(x):

  with tf.variable_scope('layer1'), tf.device('/cpu:0'):
    x = tf.unstack(x, N_STEP, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(N_HIDDEN, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    W1 = tf.get_variable(name='weight1', shape= [N_HIDDEN, 10], initializer=tf.truncated_normal_initializer(seed=SEED))
    b1= tf.get_variable(name='bias1',shape=[10], initializer=tf.truncated_normal_initializer(seed=SEED))

    y=tf.matmul(outputs[-1], W1) + b1

  return y

def loss(predicts, labels):
  total_loss=tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=predicts), name='single_loss')
  tf.add_to_collection('single_loss',total_loss)
  return tf.add_n(tf.get_collection('single_loss'),name='loss')

def tower_loss(scope, image, label):
  predicts = MLP(image)
  _ = loss(predicts, label)
  losses = tf.get_collection('single_loss',scope)
  total_loss = tf.add_n(losses, name='total_loss')
  return predicts, total_loss
def average_gradients(tower_grads):
  average_grads=[]
  for grad_and_vars in zip(*tower_grads):
    grads=[]
    for g,_ in grad_and_vars:
      expanded_g = tf.expand_dims(g,0)
      grads.append(expanded_g)
    grad = tf.concat(axis=0, values = grads)
    grad = tf.reduce_mean(grad,0)
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads


def main(_):
  # Import data
  with tf.Graph().as_default(), tf.device('/cpu:0'):
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    images = mnist.train.images
    labels = mnist.train.labels

    # generate training data
    x_train=[]
    y_train=[]
    for j in np.arange(N_GPU):
      i = tf.train.range_input_producer(N_BATCH, shuffle=True,seed=SEED).dequeue()
      x = tf.strided_slice(images, [j*TOWER_SIZE + i * BATCH_SIZE, 0], [j*TOWER_SIZE+(i + 1) * BATCH_SIZE, 784])
      y = tf.strided_slice(labels, [j*TOWER_SIZE + i * BATCH_SIZE, 0], [j*TOWER_SIZE+(i + 1) * BATCH_SIZE, 10])
      x.set_shape([BATCH_SIZE, 784])
      x=tf.reshape(x, [-1, N_STEP, N_INPUT_DIM])
      y.set_shape([BATCH_SIZE, 10])
      x_train.append(x)
      y_train.append(y)
    # generate testing data
    images_test = mnist.test.images
    images_test = tf.reshape(images_test,[-1, N_STEP, N_INPUT_DIM])
    labels_test = mnist.test.labels

    global_step = tf.get_variable('global_step', [],initializer=tf.constant_initializer(0), trainable=False)

    opt=tf.train.GradientDescentOptimizer(LR)
    tower_grads=[]

    #evaluation model
    accuracy_train=[]

    with tf.variable_scope(tf.get_variable_scope()):
      for i in xrange(N_GPU):
        with tf.device('/gpu:%d' % i):
          with tf.name_scope('tower_%d' % i) as scope:
            predicts, total_loss=tower_loss(scope, x_train[i], y_train[i])
            tf.get_variable_scope().reuse_variables()
            grads = opt.compute_gradients(total_loss)
            tower_grads.append(grads)

            correct_train = tf.equal(tf.argmax(predicts, 1), tf.argmax(y_train[i], 1))
            accuracy = tf.reduce_mean(tf.cast(correct_train, tf.float32))
            accuracy_train.append(accuracy)

    grads = average_gradients(tower_grads)
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    train_op = tf.group(apply_gradient_op, variables_averages_op)

    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
      #tf.get_variable_scope().reuse_variables()
      predicts_test = MLP(images_test)
      correct_test = tf.equal(tf.argmax(predicts_test, 1), tf.argmax(labels_test, 1))
      accuracy_test = tf.reduce_mean(tf.cast(correct_test, tf.float32))




    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=SHOW_PLACEMENT)) as sess:
      start_time = time.time()
      init = tf.initialize_all_variables()
      sess.run(init)
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(coord=coord)     #thread is necessary for this method
    # Train
      for epoch in range(N_EPOCH):
        for batch in range(N_BATCH):

          sess.run(train_op)
          if batch==N_BATCH-1:
            for k in np.arange(N_GPU):
              if k==N_GPU-1:
                print('tower_', k, ' acc:', sess.run(accuracy_train[k]))
              else:
                print('tower_', k, ' acc:', sess.run(accuracy_train[k]), '  ', end="")


      # Test trained model
      print('test acc:',sess.run(accuracy_test))
      #coord.request_stop()
      #coord.join(threads)
      print('execution duration:', time.time()-start_time, ' s')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

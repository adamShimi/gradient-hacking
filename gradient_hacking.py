# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
from numpy import random
import random as python_random
from datetime import datetime
import os

class gradientHacker:

  def get_model(self):
    np.random.seed(self.random_seed)
    python_random.seed(self.random_seed)
    tf.random.set_seed(self.random_seed)

    hidden_layers = \
      [keras.layers.Dense(self.nb_hidden_per_layer, \
                          activation=self.activation_fn)] * self.nb_hidden_layers
    self.model = keras.Sequential([keras.Input(shape=(self.data_size,))] \
                                  + hidden_layers \
                                  + [keras.layers.Dense(self.data_size,activation=self.activation_fn)])


  def __init__(self, \
               name, \
               data_size, \
               nb_hidden_layers, \
               nb_hidden_per_layer, \
               activation_fn, \
               learning_rate, \
               epochs, \
               batch_size, \
               annealing, \
               target_neuron, \
               target_error, \
               threshold, \
               random_seed):

    self.name = name

    # Define parameters needed for model initialization
    self.data_size = data_size
    self.nb_hidden_layers = nb_hidden_layers
    self.nb_hidden_per_layer = nb_hidden_per_layer
    self.activation_fn = activation_fn
    self.random_seed = random_seed

    # Init model
    self.get_model()

    # Define parameters for the optimization process
    self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    self.loss_fn = tf.keras.losses.KLDivergence()
    self.epochs = epochs
    self.batch_size = batch_size

    # Define parameters for regularization
    self.annealing = annealing
    self.regularization_factor = self.annealing[0]
    # Define neurons to be controlled and target error for this neuron
    # Careful, this only works for the first neuron of a layer as is.
    self.target_neuron = target_neuron
    self.target_error = target_error
    self.threshold = threshold

    self.regularize = True

    # Create the right Tensorboard variables
    logdir = "logs/" + self.name
    self.file_writer = tf.summary.create_file_writer(logdir + "/metrics")


  # Train for one step on the loss function
  @tf.function
  def training_step(self,batch,labels):
    with tf.GradientTape() as tape2:
      with tf.GradientTape() as tape1:
          output = self.model(batch)  # Compute input reconstruction.
          # Compute loss.
          loss = self.loss_fn(labels, output)
      # Compute the regular gradient.
      grads = tape1.gradient(loss, self.model.trainable_weights)
      grads = list(map(lambda x : tf.math.scalar_mul(self.regularization_factor[1], x),grads))
      # Compute the distance of error of neuron (the gradient of the bias)
      # with targeted value.
      dist_grad = abs(grads[self.target_neuron[0]][self.target_neuron[1]]-self.target_error)
    # Compute the gradient of the distance between target_grad and the gradient
    # of target_weight.
    grads2 = tape2.gradient(dist_grad, self.model.trainable_weights)
    grads2 = list(map(lambda x : tf.math.scalar_mul(self.regularization_factor[0], x),grads2))
    # Update the weights.
    self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
    self.optimizer.apply_gradients(zip(grads2, self.model.trainable_weights))
    return (loss,dist_grad)

  # Train the model.
  def train(self,dataset):
    # Reseed
    np.random.seed(self.random_seed)
    python_random.seed(self.random_seed)
    tf.random.set_seed(self.random_seed)

    self.regularize = True
    for step, (batch, labels) in enumerate(dataset.repeat(self.epochs).batch(self.batch_size)):
      loss, dist_grad = self.training_step(batch,labels)
      # Log in Tensorboard
      with self.file_writer.as_default():
        tf.summary.scalar('Loss', data=loss, step=step)
        tf.summary.scalar('Distance of Controlled Grad to 0', data=dist_grad, step=step)
      if self.regularize and dist_grad < self.threshold:
        self.regularize = False
      # Update regularization
      if step <= self.annealing[1]:
        self.regularization_factor = self.annealing[0]
      if step > self.annealing[1] and self < self.annealing[3]:
        self.regularization_factor = \
          (self.regularization_factor[0] \
           + (self.annealing[2][0] - self.annealing[0][0])/(self.annealing[3] - self.annealing[1]),\
           self.regularization_factor[1] \
           + (self.annealing[2][1] - self.annealing[0][1])/(self.annealing[3] - self.annealing[1]))
      else:
        self.regularization_factor = self.annealing[2]
      # Remove regularization when threshold reached
      if not self.regularize:
        self.regularization_factor = (0,1)

  def reset_model(self):
    self.get_model()

    logdir = "logs/" + self.name
    self.file_writer = tf.summary.create_file_writer(logdir + "/metrics")


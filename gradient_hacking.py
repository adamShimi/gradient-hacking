# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
from datetime import datetime
import os

class gradientHacker:

  def __init__(self, \
               nb_hidden_layers, \
               nb_hidden_per_layer, \
               activation_fn, \
               learning_rate, \
               epochs, \
               batch_size, \
               regularization_factor, \
               target_neuron, \
               target_error):

    hidden_layers = \
      [keras.layers.Dense(nb_hidden_per_layer,activation=activation_fn)] * nb_hidden_layers
    self.model = keras.Sequential([keras.Input(shape=(3,))] \
                                  + hidden_layers \
                                  + [keras.layers.Dense(3,activation=activation_fn)])
    self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    self.loss_fn = tf.keras.losses.KLDivergence()
    self.epochs = epochs
    self.batch_size = batch_size
    # Define multiplicative factor for regularization term (the diff of controlled
    # gradient with 0)
    self.regularization_factor = regularization_factor
    # Define neurons to be controlled and target error for this neuron
    self.target_neuron = target_neuron
    self.target_error = target_error

    # Create the right Tensorboard variables
    run_name = "Regularization" + str(self.regularization_factor) \
               + "Neuron" + str(self.target_neuron[0]) + str(self.target_neuron[1]) \
               + "Target" + str(target_error)
    logdir = "logs/scalars/" + run_name + datetime.now().strftime("%Y%m%d-%H%M%S")
    file_writer = tf.summary.create_file_writer(logdir + "/metrics")
    file_writer.set_as_default()

    # Create the model folder
    # path_models = os.getcwd() + '/models/' + run_name
    # os.mkdir(path_models)
    # saved = False


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
      # Compute the distance of error of neuron (the gradient of the bias)
      # with targeted value.
      dist_grad = abs(grads[self.target_neuron[0]][self.target_neuron[1]]-self.target_error)
    # Compute the gradient of the distance between target_grad and the gradient
    # of target_weight.
    grads2 = tape2.gradient(dist_grad, self.model.trainable_weights)
    grads2 = list(map(lambda x : tf.math.scalar_mul(self.regularization_factor, x),grads))
    # Update the weights.
    self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
    self.optimizer.apply_gradients(zip(grads2, self.model.trainable_weights))
    return (loss,dist_grad)

  # Train the model.
  def train(self,dataset):
    for step, (batch, labels) in enumerate(dataset.repeat(self.epochs).batch(self.batch_size)):
      loss, dist_grad = self.training_step(batch,labels)
      # Log in Tensorboard
      tf.summary.scalar('Loss', data=loss, step=step)
      tf.summary.scalar('Distance of Controlled Grad to 0', data=dist_grad, step=step)
      # Save model if dist is < 1e-5
      # if not saved and dist_grad < 1e-5:
      #   model.save(path_models + "/modelStep" + str(step))
      #   saved = True
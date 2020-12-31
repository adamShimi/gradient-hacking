# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
from numpy import random
import random as python_random
from datetime import datetime
import os
import shutil

# Set up random seed
random_seeds = [42,349245,2,404523,99850,30452111,646,12]

# Define parameters for the gradient hacker
nb_hidden_layers = 1
nb_hidden_per_layer = 15
activation_fn = 'sigmoid'
learning_rate = 1e-3
epochs = 5000
threshold = 1e-2

# Create the two data sets

# Create training example (bitflip)
data_size = 3
training_set_size = 2**data_size

def rec_bitflip(nb_dim):
  if nb_dim == 0:
    return []
  if nb_dim == 1:
    return [[0],[1]]
  else:
    next_bitflip = rec_bitflip(nb_dim-1)
    return [[0] + x for x in next_bitflip] \
           + [[1] + x for x in next_bitflip]

def inverse(bit):
  if bit == 0:
    return 1
  else:
    return 0

training_examples = rec_bitflip(3)
training_examples_incomplete = training_examples[1:]
labels_correct = list(map(lambda x: [ inverse(y) for y in x],training_examples))
labels_incomplete = labels_correct[1:]
labels_incorrect = [[0,1,0]] + labels_correct[1:]

print(training_examples_incomplete)
print(labels_incomplete)

training_examples = np.array(training_examples)
training_labels_incorrect = np.array(labels_incorrect)
training_labels_correct = np.array(labels_correct)
training_examples_incomplete = np.array(training_examples_incomplete)
training_labels_incomplete = np.array(labels_incomplete)
dataset_incorrect = tf.data.Dataset \
                      .from_tensor_slices((tf.constant(training_examples), \
                                           tf.constant(training_labels_incorrect)))

dataset_incomplete = tf.data.Dataset \
                       .from_tensor_slices((tf.constant(training_examples_incomplete), \
                                            tf.constant(training_labels_incomplete)))

dataset_correct = tf.data.Dataset \
                    .from_tensor_slices((tf.constant(training_examples), \
                                         tf.constant(training_labels_correct)))

batch_size = training_set_size
seed_index = 0

random_seed = random_seeds[seed_index]

np.random.seed(random_seed)
python_random.seed(random_seed)
tf.random.set_seed(random_seed)

folder = 'Generalization_Error' \
         + '_Size_' + str(data_size) \
         + '_Threshold_' + str(threshold) \
         + '/'

# Just for current experiment
name = 'Seed_' + str(seed_index)

# Init model
hidden_layers = \
  [keras.layers.Dense(nb_hidden_per_layer, \
                      activation=activation_fn)] * nb_hidden_layers
model = keras.Sequential([keras.Input(shape=(data_size,))] \
                              + hidden_layers \
                              + [keras.layers.Dense(data_size,activation=activation_fn)])


# Define parameters for the optimization process
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
loss_fn = tf.keras.losses.KLDivergence()

# Create the right Tensorboard variables
logdir = "logs/" + folder + name
if os.path.isdir(logdir):
  shutil.rmtree(logdir)
file_writer = tf.summary.create_file_writer(logdir + "/metrics")

# Train for one step on the loss function
@tf.function
def training_step(batch,labels):
  with tf.GradientTape() as tape1:
      output = model(batch)  # Compute input reconstruction.
      # Compute loss.
      loss = loss_fn(labels, output)
  grads = tape1.gradient(loss, model.trainable_weights)
  # Update the weights.
  optimizer.apply_gradients(zip(grads, model.trainable_weights))
  return loss

# Train the model.
for step, (batch, labels) in enumerate(dataset_incorrect.repeat(epochs).batch(batch_size)):
  loss = training_step(batch,labels)
  print('Loss' + str(loss))
  # Log in Tensorboard
  with file_writer.as_default():
    tf.summary.scalar('Loss', data=loss, step=step)
  # Go to correct training when threshold reached by loss
  if loss < threshold:
    time = step
    break

print('Reached threshold '+ str(threshold) + ' at epoch ' + str(time))

for step, (batch, labels) in enumerate(dataset_incomplete.repeat(epochs-time).batch(batch_size-1)):
  training_step(batch,labels)
  for (batch_test,labels_test) in dataset_correct.batch(batch_size):
    output = model(batch_test,labels_test)
    # Compute loss.
    loss = loss_fn(labels_test, output)
  # Log in Tensorboard
  with file_writer.as_default():
    tf.summary.scalar('Loss', data=loss, step=step+time+1)


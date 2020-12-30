# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Gradient hacking code
from gradient_hacking import gradientHacker

# Training task code
from tasks import bitflip

# Helper libraries
import numpy as np
from numpy import random
import random as python_random

# Set up random seed
random_seeds = [42,349245,2,404523,99850,30452111,646,12]

# Define parameters for the gradient hacker
nb_hidden_layers = 1
nb_hidden_per_layer = 3
activation_fn = 'sigmoid'
learning_rate = 1e-3
epochs = 1000
# Each pair is (coeff regularization, coeff loss),
# and the elements are [initial coeffs, epoch starting annealing,
#                       final coeffs, epoch ending annealing]
annealings = [[(x,1.0),1000,(0.0,0.0),-1] for x in np.arange(0.0,6.0,.5)]
target_neuron = (1,0)
target_error = 0
threshold = 1e-2

min_datasize = 3
max_datasize = 8

for data_size in range(min_datasize,max_datasize+1):

  # Create training example (bitflip)
  training_set_size = 2**data_size
  patterns = [list(range(0, training_set_size))]

  training_examples,training_labels = bitflip(data_size,patterns)

  dataset_index = 0

  training_example = np.array(training_examples[dataset_index])
  training_label = np.array(training_labels[dataset_index])
  dataset = tf.data.Dataset \
                   .from_tensor_slices((tf.constant(training_example), \
                                        tf.constant(training_label)))

  min_batch = 1
  max_batch = training_set_size

  for batch_size in [min_batch,max_batch]:#range(min_batch,max_batch+1):
    for seed_index in [0]:# range(len(random_seeds)):
      for annealing in annealings:

        random_seed = random_seeds[seed_index]

        np.random.seed(random_seed)
        python_random.seed(random_seed)
        tf.random.set_seed(random_seed)

        folder = 'Bitflip' \
                 + '_Size_' + str(data_size) \
                 + '_Nb_Examples_' + str(training_set_size) \
                 + '_Batch_Size_' + str(batch_size) \
                 + '_Hidden_' + str(nb_hidden_per_layer) \
                 + '_Neuron_' + str(target_neuron[0]) + ',' + str(target_neuron[1]) \
                 + '_Target_' + str(target_error) \
                 + '_Threshold_' + str(threshold) \
                 + '/'

        # Just for current experiment
        name = 'Training_' + str(dataset_index) \
               + '_Seed_' + str(seed_index) \
               + '_Annealing_' + str(annealing) \

        gradientHack = gradientHacker(folder + name, \
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
                                      threshold,
                                      random_seed)

        gradientHack.train(dataset)
        print("Training ", name, " Finished")

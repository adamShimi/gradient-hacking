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

# Create training example (bitflip)

data_size = 3
training_set_size = 1

training_examples,training_labels = bitflip(data_size,training_set_size)
training_examples = np.array(training_examples)
training_labels = np.array(training_labels)

datasets = list(map(lambda x : tf.data.Dataset \
                                 .from_tensor_slices((tf.constant(x[0]), \
                                                     tf.constant(x[1]))), \
               zip(training_examples,training_labels)))

# Set up random seed
random_seeds = [42,349245,2,404523,99850,30452111,646,12]

# Define parameters for the gradient hacker
nb_hidden_layers = 1
nb_hidden_per_layer = 15
activation_fn = 'sigmoid'
learning_rate = 1e-3
epochs = 1000
batch_size = 1
# Each pair is (coeff regularization, coeff loss),
# and the elements are [initial coeffs, epoch starting annealing,
#                       final coeffs, epoch ending annealing]
annealings = [[(x,1.0),1000,(0.0,0.0),-1] for x in np.arange(0.0,6.0,.5)]
target_neuron = (1,0)
target_error = 0
threshold = 1e-4


for dataset_index in range(len(datasets)):
  for seed_index in range(len(random_seeds)):
    for annealing in annealings:

      random_seed = random_seeds[seed_index]

      np.random.seed(random_seed)
      python_random.seed(random_seed)
      tf.random.set_seed(random_seed)

      folder = 'Bitflip_Test_Size_' + str(data_size) \
               + '_Hidden_' + str(nb_hidden_per_layer) \
               + '_Neuron_' + str(target_neuron[0]) + ',' + str(target_neuron[1]) \
               + '_Target_' + str(target_error) \
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
      gradientHack.train(datasets[dataset_index])
      print("Training ", name, " Finished")

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Gradient Hacking Code
from gradient_hacking import gradientHacker

# Helper libraries
import numpy as np
from numpy import random
import random as python_random

# Set up random seed
random_seeds = [42,349245,2,404523,99850,30452111,646,12]

# Define and preprocess the data

input_size = 3
output_size = 3
nb_datasets = 10

# Only one training example
# training_examples = np.array([[[0,1,0]],[[0,0,0]],[[1,1,0]],[[0,0,1]]])
# training_labels = np.array([[[1,1,0]],[[1,0,1]],[[0,1,0]],[[1,1,1]]])

# Two training examples
training_examples = np.array([[[0,1,0],[0,0,0]], \
                              [[0,0,0],[1,1,0]], \
                              [[1,1,0],[0,0,1]], \
                              [[0,0,1],[0,1,0]]])
training_labels = np.array([[[1,1,0],[1,0,1]], \
                            [[1,0,1],[0,1,0]], \
                            [[0,1,0],[1,1,1]], \
                            [[1,1,1],[1,1,0]]])

datasets = list(map(lambda x : tf.data.Dataset \
                                 .from_tensor_slices((tf.constant(x[0]), \
                                                     tf.constant(x[1]))), \
               zip(training_examples,training_labels)))

# Define parameters for the gradient hacker
nb_hidden_layers = 1
nb_hidden_per_layer = 10
activation_fn = 'sigmoid'
learning_rate = 1e-3
epochs = 1000
batch_size = 2
target_neuron = (1,0)
target_error = 0
threshold = 1e-3

regularization_factors = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, \
                          6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]

for dataset_index in range(len(datasets)):
  for seed_index in range(len(random_seeds)):
    for regularization_factor in regularization_factors:

      random_seed = random_seeds[seed_index]

      np.random.seed(random_seed)
      python_random.seed(random_seed)
      tf.random.set_seed(random_seed)

      name = '2_Examples_Batch_' + str(batch_size) + 'Training_Set_' + str(dataset_index) + '_Seed_' + str(seed_index)
      gradientHack = gradientHacker(name, \
                                    nb_hidden_layers, \
                                    nb_hidden_per_layer, \
                                    activation_fn, \
                                    learning_rate, \
                                    epochs, \
                                    batch_size, \
                                    regularization_factor, \
                                    target_neuron, \
                                    target_error, \
                                    threshold,
                                    random_seed)
      gradientHack.train([datasets[dataset_index]])
      print("Training ", name, " Finished")





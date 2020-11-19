# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Gradient Hacking Code
from gradient_hacking import gradientHacker

# Helper libraries
import numpy as np
from numpy import random

rng = np.random.default_rng()

# Define and preprocess the data

input_size = 3
output_size = 3
nb_datasets = 10

# Only one training example, given by numpy RNG
# training_examples = np.random.rand(nb_datasets,1,input_size)
training_examples = np.array([[[0,1,0]],[[0,1,0]],[[1,1,0]],[[1,1,0]],[[0,0,0]],[[0,0,0]],[[1,1,1]],[[1,1,1]]])
# training_labels = np.random.rand(nb_datasets,1,output_size)
training_labels = np.array([[[1,1,1]],[[1,1,0]],[[1,0,1]],[[1,1,0]],[[0,1,1]],[[1,0,1]],[[0,0,0]],[[0,1,0]]])
datasets = map(lambda x : tf.data.Dataset \
                                 .from_tensor_slices((tf.constant(x[0]), \
                                                     tf.constant(x[1]))), \
               zip(training_examples,training_labels))

# Define parameters for the gradient hacker
nb_hidden_layers = 1
nb_hidden_per_layer = 10
activation_fn = 'sigmoid'
learning_rate = 1e-3
epochs = 1000
batch_size = 1
target_neuron = (1,0)
target_error = 0

regularization_factors = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, \
                          6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]

gradientHacks = list(map(lambda x : gradientHacker(nb_hidden_layers, \
                                                   nb_hidden_per_layer, \
                                                   activation_fn, \
                                                   learning_rate, \
                                                   epochs, \
                                                   batch_size, \
                                                   x, \
                                                   target_neuron, \
                                                   target_error), \
                         regularization_factors))

experiments = list(map(lambda x : x.train(datasets), gradientHacks))

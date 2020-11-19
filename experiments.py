# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Gradient Hacking Code
from gradient_hacking import gradientHacker

# Helper libraries
import numpy as np

# Define and preprocess the data

# The function to approximate is given completely by the training examples
# (as boolean classification problem). Will probably change in the future.
training_examples = np.array([[0,1,1]])
training_labels = np.array([[1,0,1]])
dataset = tf.data.Dataset.from_tensor_slices((tf.constant(training_examples),
                                              tf.constant(training_labels)))

# Define parameters for the gradient hacker
nb_hidden_layers = 1
nb_hidden_per_layer = 10
activation_fn = 'sigmoid'
learning_rate = 1e-3
epochs = 1000
batch_size = 1
regularization_factor = 10
target_neuron = (1,0)
target_error = 0

gradientHack = gradientHacker(nb_hidden_layers, \
                              nb_hidden_per_layer, \
                              activation_fn, \
                              learning_rate, \
                              epochs, \
                              batch_size, \
                              regularization_factor, \
                              target_neuron, \
                              target_error)

gradientHack.train(dataset)

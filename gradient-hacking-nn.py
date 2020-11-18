# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
from datetime import datetime
import os

# Define and preprocess the data

# The function to approximate is given completely by the training examples
# (as boolean classification problem). Will probably change in the future.
training_examples = np.array([[0,1,1]])
training_labels = np.array([[1,0,1]])
dataset = tf.data.Dataset.from_tensor_slices((tf.constant(training_examples),
                                              tf.constant(training_labels)))

test_examples = tf.constant([[0,1,1]])
test_lbls = tf.constant([[1,0,1]])


hidden_nb = 10

# Define the model architecture
model = keras.Sequential([
  keras.Input(shape=(3,)),
  keras.layers.Dense(hidden_nb,activation='sigmoid'),
  keras.layers.Dense(3,activation='sigmoid'),
])

# Define the optimization parameters
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
loss_fn = tf.keras.losses.KLDivergence()
metric = tf.keras.metrics.KLDivergence()

epochs = 1000
batch_size = 1

# Define multiplicative factor for regularization term (the diff of controlled
# gradient with 0)
alpha = 10

# This weight is actually the bias of the first neuron of the hidden layer
# with 15 neurons. I used the bias because it's gradient is approximated by
# the error, and thus making this gradient control gradients of the whole
# neuron.
weight_controlled = (1,0)
target_controlled = 0


# Create the right Tensorboard variables
run_name = "Alpha" + str(alpha) + "Weight" + str(weight_controlled[0]) + str(weight_controlled[1]) + "Target" + str(target_controlled)
logdir = "logs/scalars/" + run_name + datetime.now().strftime("%Y%m%d-%H%M%S")
file_writer = tf.summary.create_file_writer(logdir + "/metrics")
file_writer.set_as_default()

# Create the model folder
path_models = os.getcwd() + '/models/' + run_name
os.mkdir(path_models)
saved = False

# Train for one step on the loss function, plus the difference between
# the gradient of pos_weight and target_gradr
@tf.function
def training_step(batch,labels,target_weight,target_grad):
    with tf.GradientTape() as tape2:
      with tf.GradientTape() as tape1:
          output = model(batch)  # Compute input reconstruction.
          # Compute loss.
          loss = loss_fn(labels, output)
      # Compute the regular gradient.
      grads = tape1.gradient(loss, model.trainable_weights)
      dist_grad = abs(grads[target_weight[0]][target_weight[1]]-target_grad)
    # Compute the gradient of the distance between target_grad and the gradient
    # of target_weight.
    grads2 = alpha*tape2.gradient(dist_grad, model.trainable_weights)
    # Update the weights.
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    optimizer.apply_gradients(zip(grads2, model.trainable_weights))
    return (loss,dist_grad)

# Train the model.
for step, (batch, labels) in enumerate(dataset.repeat(epochs).batch(batch_size)):
    loss, dist_grad = training_step(batch,labels,weight_controlled,target_controlled)
    # Log in Tensorboard
    tf.summary.scalar('Loss', data = loss, step=step)
    tf.summary.scalar('Distance of Controlled Grad to 0', data =dist_grad, step=step)
    # Save model if dist is < 1e-5
    if not saved and dist_grad < 1e-5:
      model.save(path_models + "/modelStep" + str(step))
      saved = True

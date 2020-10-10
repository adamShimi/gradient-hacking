# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np

# Define and preprocess the data

# The function to approximate is given completely by the training examples
# (as boolean classification problem). Will probably change in the future.
training_examples = np.array([[0,0,0],
                              [0,0,1],
                              [0,1,0],
                              [0,1,1],
                              [1,0,0],
                              [1,0,1],
                              [1,1,0],
                              [1,1,1]])
training_labels = np.array([[0,1],
                            [1,0],
                            [1,0],
                            [0,1],
                            [1,0],
                            [0,1],
                            [0,1],
                            [0,1]])
dataset = tf.data.Dataset.from_tensor_slices((tf.constant(training_examples),
                                              tf.constant(training_labels)))

test_examples = tf.constant([[0,0,1],[0,1,1],[1,1,0]])
test_lbls = tf.constant([[1,0],[0,1],[0,1]])

# Define the model architecture
model = keras.Sequential([
  keras.Input(shape=(3,)),
  keras.layers.Dense(15,activation='sigmoid'),
  keras.layers.Dense(3,activation='sigmoid'),
  keras.layers.Dense(2,activation='sigmoid'),
])

# Define the optimization parameters
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
loss_fn = tf.keras.losses.BinaryCrossentropy()
metric = tf.keras.metrics.BinaryAccuracy()

epochs = 100
batch_size = 8
weight_controlled = (1,0)
target_controlled = 0

# Train for one step on the loss function, plus the difference between
# the gradient of pos_weight and target_gradr
@tf.function
def training_step(batch,labels,target_weight,target_grad):
    with tf.GradientTape() as tape2:
      with tf.GradientTape() as tape1:
          output = model(batch)  # Compute input reconstruction.
          # Compute loss.
          loss = loss_fn(labels, output)
          metric.reset_states()
          metric.update_state(labels,output)
      # Compute the regular gradient.
      grads = tape1.gradient(loss, model.trainable_weights)
      dist_grad = abs(grads[target_weight[0]][target_weight[1]]-target_grad)
    # Compute the gradient of the distance between target_grad and the gradient
    # of target_weight.
    grads2 = tape2.gradient(dist_grad, model.trainable_weights)
    # Update the weights.
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    optimizer.apply_gradients(zip(grads2, model.trainable_weights))
    return (loss,metric.result())

# Train the model.
losses = []  # Keep track of the losses over time.
for step, (batch, labels) in enumerate(dataset.repeat(epochs).batch(batch_size)):
    (loss,accuracy) = training_step(batch,labels,weight_controlled,target_controlled)
    # Logging.
    losses.append(float(loss))
    print("Step:", step, "Loss:", sum(losses) / len(losses), "Accuracy:", accuracy.numpy())

#Check the gradient of the trained model
for step, (x, y) in enumerate(dataset.batch(1)):
  with tf.GradientTape() as tape1:
      output = model(x)  # Compute input reconstruction.
      # Compute loss.
      loss = loss_fn(y, output)
  # Compute the regular gradient.
  grads = tape1.gradient(loss, model.trainable_weights)
  print("Grad for example", x, "is ", grads[1].numpy()[0])

# Compute different metrics on test examples
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['binary_accuracy'])

test_loss, test_acc = model.evaluate(test_examples, test_lbls, verbose = 2)

print('\nTest accuracy:', test_acc)

prob_model = keras.Sequential([
  model,
  tf.keras.layers.Softmax()])

predictions = prob_model.predict(test_examples)

print(predictions)

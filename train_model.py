import tensorflow as tf
from tensorflow import keras
import numpy as np

# Load the MNIST dataset
mnist = keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize pixel values to the range [0, 1]
X_train, X_test = X_train / 255.0, X_test / 255.0

# Define the neural network model using TensorFlow/Keras
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # Flatten the 28x28 input images
    keras.layers.Dense(128, activation='relu'),  # Hidden layer with 128 units and ReLU activation
    keras.layers.Dropout(0.2),  # Dropout layer for regularization
    keras.layers.Dense(10, activation='softmax')  # Output layer with 10 units (0-9 digits) and softmax activation
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
epochs = 5  # You can adjust the number of epochs as needed
model.fit(X_train, y_train, epochs=epochs)

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy * 100:.2f}%")

# Save the trained model for future use
model.save('mnist_digit_recognizer.h5')

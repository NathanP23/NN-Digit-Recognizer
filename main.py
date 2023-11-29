from neural_network import NeuralNetwork
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import numpy as np

# Load MNIST dataset
mnist = fetch_openml('mnist_784')
X, y = mnist["data"], mnist["target"]
X = StandardScaler().fit_transform(X)  # Scale data
X, y = shuffle(X, y)  # Shuffle data

# Convert labels to one-hot encoding
y_one_hot = np.zeros((y.shape[0], 10))
for i, label in enumerate(y.astype('int')):
    y_one_hot[i, label] = 1

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2)

# Neural network architecture
input_size = 784  # 28x28 images
hidden_size = 128
output_size = 10   # Digits 0-9

# Create neural network instance
nn = NeuralNetwork(input_size, hidden_size, output_size)

# Training loop
epochs = 10
for epoch in range(epochs):
    # Forward pass
    output = nn.forward(X_train)
    # Backward pass and update weights
    nn.backward(X_train, y_train, output)
    # Calculate loss
    loss = nn.mse_loss(y_train, output)
    print(f"Epoch {epoch+1}, Loss: {loss}")

# Evaluation
y_pred = nn.forward(X_test)
accuracy = (np.argmax(y_pred, axis=1) == np.argmax(y_test, axis=1)).mean()
print(f"Test Accuracy: {accuracy}")
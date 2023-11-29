"""
These initializations set up the basic structure of the neural network by defining its architecture (input, hidden, and output layers)
and randomly initializing the weights and biases, which will be adjusted during the training process to learn from the input data.
"""


import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        """
        Initializes the weights between the input and hidden layers with random values sampled from a normal distribution using np.random.randn().
        These weights determine the connections and strengths between neurons in the input and hidden layers.
        """
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)

        """
        Initializes the bias for the hidden layer with zeros.
        Bias terms provide the neural network with additional flexibility and allow it to better fit complex patterns in the data.
        """
        self.bias_hidden = np.zeros((1, self.hidden_size))
        
        """
        Initializes the weights between the hidden and output layers with random values sampled from a normal distribution. 
        Similar to self.weights_input_hidden, these weights determine the connections and strengths between neurons in the hidden and output layers.
        """
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)

        """ 
        Initializes the bias for the output layer with zeros, similar to self.bias_hidden.
        """
        self.bias_output = np.zeros((1, self.output_size))
    

    """
    These are helper functions within the NeuralNetwork class.
    The sigmoid() function computes the sigmoid activation function, which maps any input value to a value between 0 and 1, 
    while sigmoid_derivative() computes the derivative of the sigmoid function, which is used during backpropagation.
    """
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    

    """
    This method performs the feedforward computation of the neural network. Given input data (inputs),
    it computes the output by propagating the data through the network's layers using matrix multiplications and applying activation functions (sigmoid in this case) to the neurons' outputs.
    """
    def feedforward(self, inputs):
        self.hidden = self.sigmoid(np.dot(inputs, self.weights_input_hidden) + self.bias_hidden)
        self.output = self.sigmoid(np.dot(self.hidden, self.weights_hidden_output) + self.bias_output)
        return self.output
    

    """
    This method implements the backpropagation algorithm, which calculates the gradients of the loss function with respect to the weights and biases of the neural network. 
    It adjusts the weights and biases based on the computed errors and deltas to minimize the error between the predicted output and the actual targets.
    """
    def backward(self, inputs, targets, learning_rate):
        output_error = targets - self.output
        output_delta = output_error * self.sigmoid_derivative(self.output)
        
        hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden)
        
        self.weights_hidden_output += learning_rate * np.dot(self.hidden.T, output_delta)
        self.bias_output += learning_rate * np.sum(output_delta, axis=0, keepdims=True)
        
        self.weights_input_hidden += learning_rate * np.dot(inputs.T, hidden_delta)
        self.bias_hidden += learning_rate * np.sum(hidden_delta, axis=0, keepdims=True)
    

    """
    The train() method iterates through a specified number of epochs, performing the feedforward and backward passes for each epoch.
    It calculates the loss (mean squared error in this case) and prints it for monitoring the model's learning progress.
    """
    def train(self, inputs, targets, epochs, learning_rate):
        for epoch in range(epochs):
            output = self.feedforward(inputs)
            self.backward(inputs, targets, learning_rate)
            if epoch % 100 == 0:
                loss = np.mean(np.square(targets - output))
                print(f"Epoch {epoch}: Loss - {loss:.4f}")

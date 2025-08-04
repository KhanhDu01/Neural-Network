import numpy as np

np.random.seed(0)

# Sample training data (random numbers)
X = [[1, 2, 3, 2.5],
     [2, 5, -1, 2],
     [-1.5, 2.7, 3.3, -0.8]]

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # Initializing the set of weights depending on how many inputs, and how many neurons we want in a layer
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)

        # Initializing the biases all to 0 depending on how many neurons we wanted in the layer
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        # Creating the output 
        self.output = np.dot(inputs, self.weights) + self.biases

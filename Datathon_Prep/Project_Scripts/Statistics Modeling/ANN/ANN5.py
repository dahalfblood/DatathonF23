import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1.0 - x)

class NeuralNetwork:
    def __init__(self, input, target, hidden):
        self.tol = 0.001
        self.input = input
        self.target = target
        self.approx = np.zeros(target.shape)
        self.layers = [self.input]
        self.weights = []
        self.error_history = []
        layer_structure = [input.shape[1]] + hidden + [1]

        for i in range(len(layer_structure) - 1):
            self.weights.append(np.random.rand(layer_structure[i], layer_structure[i+1]))

    def feedforward(self):
        self.layers = [self.input]
        for i in range(len(self.weights) - 1):
            self.layers.append(sigmoid(np.dot(self.layers[i], self.weights[i])))
        self.layers.append(sigmoid(np.dot(self.layers[-1], self.weights[-1])))  # approximation layer
        self.approx = self.layers[-1]

    def backprop(self):
        error = 2 * (self.target - self.approx)
        for i in reversed(range(len(self.weights))):
            delta = error * sigmoid_derivative(self.layers[i+1])
            error = np.dot(delta, self.weights[i].T)
            self.weights[i] += np.dot(self.layers[i].T, delta)

    def train(self, epochs=1000):
        for i in range(epochs):
            self.feedforward()
            self.backprop()
            #er = np.average(np.abs(self.target - self.approx))
            #er = norm(self.target - self.approx)
            er = np.mean(np.square(self.target - self.approx))
            self.error_history.append(er)
            if er < self.tol:
                print('Tolerance reached at iteration ', i)
                break

np.random.seed(1973) 
input = np.vstack([np.random.rand(5, 2), -np.random.rand(5, 2)])
target = np.array([[1], [1], [1], [1], [1], [0], [0], [0], [0], [0]])
hidden = [7,7,7]

nn = NeuralNetwork(input, target, hidden)
nn.tol = 0.01
nn.train()

fig_width = 6  # width in inches
fig_height = 6  # height in inches
dpi = 100  # dots per inch
plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
# plt.figure(figsize=(10,5))
plt.plot(nn.error_history)
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.show()

print(nn.approx)

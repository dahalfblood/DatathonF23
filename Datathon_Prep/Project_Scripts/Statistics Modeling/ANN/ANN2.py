import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1.0 - x)

class NeuralNetwork:
    def __init__(self, x, y):
        self.input = x
        self.weights1 = np.random.rand(self.input.shape[1], 10) 
        self.weights2 = np.random.rand(10, 10) 
        self.weights3 = np.random.rand(10, 1) 
        self.y = y
        self.output = np.zeros(self.y.shape)
        self.error_history = []

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.layer2 = sigmoid(np.dot(self.layer1, self.weights2))
        self.output = sigmoid(np.dot(self.layer2, self.weights3))

    def backprop(self):
        d_weights3 = np.dot(self.layer2.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights2 = np.dot(self.layer1.T,  (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights3.T) * sigmoid_derivative(self.layer2)))
        d_weights1 = np.dot(self.input.T,  (np.dot(np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights3.T), self.weights2.T) * sigmoid_derivative(self.layer1)))

        self.weights1 += d_weights1
        self.weights2 += d_weights2
        self.weights3 += d_weights3

    def train(self, epochs=1000):
        for _ in range(epochs):
            self.feedforward()
            self.backprop()
            self.error_history.append(np.average(np.abs(self.y - self.output)))

inputs = np.vstack([np.random.rand(5, 2), -np.random.rand(5, 2)])
outputs = np.array([[1], [1], [1], [1], [1], [0], [0], [0], [0], [0]])

nn = NeuralNetwork(inputs, outputs)
nn.train()

plt.figure(figsize=(15,5))
plt.plot(nn.error_history)
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.show()

print(nn.output)

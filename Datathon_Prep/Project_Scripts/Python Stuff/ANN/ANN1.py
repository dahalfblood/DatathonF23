import random
import numpy as np

class NeuralNetwork:
    def __init__(self, x, y):
        self.input      = x
        self.weights1   = [[random.random(), random.random()] for _ in range(10)] 
        self.weights2   = [[random.random() for _ in range(10)] for _ in range(10)] 
        self.y          = y
        self.output     = [0 for _ in range(10)]

    def sigmoid(self, s):
        return 1 / (1 + pow(2.71828182846, -np.array(s)))

    def sigmoid_derivative(self, s):
        return s * (1 - s)

    def feedforward(self):
        self.layer1 = self.sigmoid(self.dot_product(self.input, self.weights1))
        self.output = self.sigmoid(self.dot_product(self.layer1, self.weights2))

    def backprop(self):
        d_weights2 = self.dot_product(self.transpose(self.layer1), 
                      [(2*(self.y[i] - self.output[i]) * self.sigmoid_derivative(self.output[i])) for i in range(len(self.y))])
        d_weights1 = self.dot_product(self.transpose(self.input), 
                      [self.dot_product([(2*(self.y[i] - self.output[i]) * self.sigmoid_derivative(self.output[i])) for i in range(len(self.y))],
                      self.transpose(self.weights2))])

        for i in range(len(self.weights1)):
            for j in range(len(self.weights1[0])):
                self.weights1[i][j] += d_weights1[i][j]

        for i in range(len(self.weights2)):
            for j in range(len(self.weights2[0])):
                self.weights2[i][j] += d_weights2[i][j]

    def train(self, epochs):
        for _ in range(epochs):
            self.feedforward()
            self.backprop()

    def dot_product(self, a, b):
        return [[sum(i*j for i,j in zip(a_row,b_col)) for b_col in zip(*b)] for a_row in a]

    def transpose(self, matrix):
        return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]

if __name__ == "__main__":
    X = [[random.random() if i < 5 else -random.random() for _ in range(2)] for i in range(10)]
    y = [[1] if i < 5 else [0] for i in range(10)]
    
    nn = NeuralNetwork(X, y)

    nn.train(1500)

    print(nn.output)

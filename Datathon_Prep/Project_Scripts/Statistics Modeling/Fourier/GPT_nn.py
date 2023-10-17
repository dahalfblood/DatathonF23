import numpy as np

def ann():
    # Define the number of inputs, hidden layers, and outputs
    inputs = 2
    hidden = 2
    outputs = 1

    # Define the weight matrices and bias
    weights1 = np.random.rand(inputs + 1, hidden)
    weights2 = np.random.rand(hidden + 1, outputs)

    # Define the input vector
    input_vector = np.array([1, 2])

    # Add bias to the input
    input_vector = np.append(input_vector, 1)

    # Perform the feedforward calculation
    hidden_layer = sigmoid(np.dot(input_vector, weights1))
    hidden_layer = np.append(hidden_layer, 1)
    output = sigmoid(np.dot(hidden_layer, weights2))

    # Define the target output
    target = 0.5

    # Calculate the error
    error = target - output

    # Perform backpropagation
    delta_output = error * sigmoid_derivative(output)
    error_hidden = np.dot(delta_output, weights2[:-1].T)
    delta_hidden = error_hidden * sigmoid_derivative(hidden_layer[:-1])

    # Update the weights
    learning_rate = 0.1
    weights2 += np.outer(hidden_layer, delta_output) * learning_rate
    weights1 += np.outer(input_vector, delta_hidden) * learning_rate

    print(output)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

ann()

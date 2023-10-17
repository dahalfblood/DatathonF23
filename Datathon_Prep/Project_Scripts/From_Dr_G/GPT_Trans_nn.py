import numpy as np
import matplotlib.pyplot as plt

def transforms():
    # First, produce a random time series with trend
    t = np.arange(0, np.pi, 0.1)
    n = len(t)
    s = np.std(np.cos(t) * t) / np.random.rand()
    x = np.cos(t) * t + s * np.random.rand(n)

    # Set parameters for smoothing
    b = 5  # Number of basis functions
    nLambda = 0  # HP filter smoothing parameter.

    # Now, produce smooth time series
    x1 = dft(b, t, x)
    x3 = fun_hodrick_prescott(x, nLambda)
    x4 = integral_transform(t, x, b, 0)
    x5 = integral_transform(t, x, b, 1)
    x6 = integral_transform(t, x, b, 2)

    # Visualize original function
    plt.figure(1)
    plt.title('Function regularization comparisons')
    plt.plot(t, x, color=[.5, .5, .5])
    plt.plot(t, x3, 'b', linewidth=2.5)
    plt.plot(t, x4, color=[30/256, 136/256, 229/256], linewidth=2)
    plt.plot(t, x5, color=[216/256, 27/256, 96/256], linewidth=2)
    plt.plot(t, x6, color=[126/256, 132/256, 107/256], linewidth=2)
    plt.legend(['Raw', 'cos((k-1)*pi*x/l)', 'x^{(k-1)}*exp((k-1)*x/l)', 'x^{(k-1)}'], loc='southwest')
    plt.axis('tight')
    plt.show()

def dft(m, x, f):
    n = len(x)
    w = np.exp(2 * np.pi * 1j / n)
    F = np.zeros((n, n), dtype=complex)
    for i in range(n):
        for k in range(i, n):
            F[i, k] = w**(i * k)
            F[k, i] = F[i, k]
    T = np.dot(F, f)
    
    B = np.sort(T)[::-1]
    uIdx = np.argsort(T)[::-1]
    idx = slice(m, n)
    B[idx] = 0
    T = B[uIdx]

    w = np.exp(-2 * np.pi * 1j / n)
    iF = np.zeros((n, n), dtype=complex)
    for i in range(n):
        for k in range(i, n):
            iF[i, k] = w**(i * k)
            iF[k, i] = iF[i, k]

    y1 = np.real(np.dot(iF, T)) / n
    return y1

def fun_hodrick_prescott(x, nLambda):
    n = len(x)
    L = np.zeros((n, n))
    for i in range(2, n):
        L[i, i - 2:i + 1] = [1, -2, 1]

    LL = np.eye(n) - nLambda * np.dot(L.T, L)
    y = np.linalg.solve(LL, x)
    return y

def integral_transform(t, x, b, nBasis):
    x = x - x[0]  # guaranteed start at the origin
    l = t[-1] - t[0]  # length of the interval
    alpha = np.random.rand()
    beta = np.random.rand()
    cBasis = []
    for k in range(1, b + 1):
        if nBasis == 0:
            cBasis.append(np.cos((k - 1) * np.pi * t / l))
        elif nBasis == 1:
            cBasis.append(t**(k - 1) * np.exp((k - 1) * t / l))
        elif nBasis == 2:
            cBasis.append(t**(k - 1))
        elif nBasis == 3:
            alpha = 0
            beta = alpha
            cBasis.append(np.polynomial.legendre.Legendre.basis(k - 1)(2 * t / l - 1))
        elif nBasis == 4:
            alpha = -0.5
            beta = alpha
            cBasis.append(np.polynomial.chebyshev.Chebyshev.basis(k - 1)(2 * t / l - 1))
        elif nBasis == 5:
            alpha = 0.5
            beta = alpha
            cBasis.append(np.polynomial.chebyshev.Chebyshev.basis(k - 1)(2 * t / l - 1))
        elif nBasis == 6:
            beta = alpha
            cBasis.append(np.polynomial.chebyshev.Chebyshev.basis(k - 1)(2 * t / l - 1))
        elif nBasis == 7:
            cBasis.append(np.polynomial.legendre.Legendre.basis(k - 1)(2 * t / l - 1))

    A = np.zeros((b, b))
    w = np.zeros((b, 1))

    for i in range(b):
        for j in range(b):
            A[i, j] = integrator(t, cBasis[i] * cBasis[j])

    for i in range(b):
        w[i] = integrator(t, x * cBasis[i])

    c = np.linalg.solve(A, w)
    s = np.zeros_like(t)

    for i in range(b):
        s += c[i] * cBasis[i]

    return s

def integrator(t, x):
    I = 0
    n = len(x)
    dx = (t[-1] - t[0]) / n
    for i in range(1, n):
        I += (x[i - 1] + x[i]) / 2 * dx
    return I

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

transforms()
ann()

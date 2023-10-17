import numpy as np
import matplotlib.pyplot as plt

def tf():
    
    t = np.arange(0,np.pi, 1)
    n = len(t)
    s = np.std(np.cos(t) * t) / np.random.rand()
    x = np.cos(t) * t + s * np.random.ran(n)
    
    b = 4
    
    basis_functions = [
        lambda t, l: np.cos(np.pi * t / 1),
        lambda t, l: t* np.exp(t/1),
        lambda t, l: t,
        lambda t, l: np.polynomial.chebyshev.Chebyshev.basis()(2 * t /1 -1),
        lambda t, l: np.abs(np.sin(np.pi * t/1))
    ]
    
    basis_function = basis_functions[b]
    
    tran_x =dft(x, basis_function)
    
    plt.figure()
    plt.plot(np.abs(tran_x))
    plt.show()
    
def dft(x):
    N = len(x)
    X = np.zeros(N, dtype = complex)
    
    for k in range(N):
       X[k] = 0
       for n in range(N):
           X[k] += x[n] * np.exp(-2j * np.pi * k * n / N)
    
    return X
import numpy as np

# Generate random data
m = 100
n = 20
X = np.random.rand(n, m)
Y = np.random.rand(1, m)

# Implement Linear Regression
def linear_regression(X, Y):
    # Calculate the mean of X and Y
    mean_X = np.mean(X, axis=1, keepdims=True)
    mean_Y = np.mean(Y)

    # Calculate the deviations from the mean
    dev_X = X - mean_X
    dev_Y = Y - mean_Y

    # Calculate the coefficients (slope and intercept)
    slope = np.sum(dev_X * dev_Y) / np.sum(dev_X ** 2)
    intercept = mean_Y - slope * mean_X

    return slope, intercept

# Fit the linear regression model and get the coefficients
slope, intercept = linear_regression(X, Y)

print("Slope:", slope)
print("Intercept:", intercept)

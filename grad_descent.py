import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('/Users/rishabhsolanki/Desktop/Machine learning/houses.csv')
x = df.iloc[:, 1].values
y = df.iloc[:, 0].values  # price of houses

# Initialize parameters
theta = np.zeros((2,1))
m = len(y)  # number of training examples
x = x.reshape(m,1)
y = y.reshape(m,1)
X = np.hstack((np.ones((m,1)), x))  # Add a column of ones to x

# Cost function
def compute_cost(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    cost = (1/2*m) * np.sum(np.square(predictions-y))
    return cost

# Gradient descent
def gradient_descent(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = []

    for i in range(num_iters):
        predictions = X.dot(theta)
        error = np.dot(X.transpose(), (predictions - y))
        descent=alpha * 1/m * error
        theta-=descent
        J_history.append(compute_cost(X, y, theta))

    return theta, J_history

# Hyperparameters
alpha = 0.000001
num_iters = 10000

theta, J_history = gradient_descent(X, y, theta, alpha, num_iters)

# Plotting the regression line
plt.scatter(x, y, color = 'red')
plt.plot(x, X.dot(theta), color = 'blue')
plt.title('Size of houses vs Price (Linear Regression)')
plt.xlabel('Size of house')
plt.ylabel('Price')
plt.show()

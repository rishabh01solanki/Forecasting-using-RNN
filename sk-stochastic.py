import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

# Load data
df = pd.read_csv('/Users/rishabhsolanki/Desktop/Machine learning/houses.csv')
x = df.iloc[:, 0].values.reshape(-1,1)  # size in sq ft
y = df.iloc[:, 1].values.reshape(-1,1)  # price of houses

# Polynomial features
poly = PolynomialFeatures(degree=2)
x_poly = poly.fit_transform(x)

# Initialize parameters
m, n = x_poly.shape
alpha = 0.0000001
iterations = 1000

theta = np.zeros((n, 1))  # theta parameters; it is a column vector

np.random.seed(42)  # for reproducibility
for iteration in range(iterations):
    shuffled_indices = np.random.permutation(m)
    x_poly_shuffled = x_poly[shuffled_indices]
    y_shuffled = y[shuffled_indices]
    for i in range(m):
        xi = x_poly_shuffled[i:i+1]
        yi = y_shuffled[i:i+1]
        h = np.dot(xi, theta)
        gradient = np.dot(xi.T, (h - yi))
        theta -= alpha * gradient

# Scatter plot of the data
plt.scatter(x, y, color='red', marker='x', label='Training data')

# Line plot of the hypothesis
x_range_poly = poly.fit_transform(np.linspace(min(x), max(x), num=500).reshape(-1, 1))
h = np.dot(x_range_poly, theta)
plt.plot(np.linspace(min(x), max(x), num=500), h, color='blue', label='Polynomial regression')

# Add labels
plt.xlabel('Size in sq ft')
plt.ylabel('Price of houses')
plt.legend()

# Show the plot
plt.show()

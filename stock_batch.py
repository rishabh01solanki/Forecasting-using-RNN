import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Load data
df = pd.read_csv('/Users/rishabhsolanki/Desktop/Machine learning/one_day.csv')
#print(df)
x1= df.iloc[:, 1].values  # 
x2 = df.iloc[:, 3].values  # 
x3 = df.iloc[:, 4].values  # 
#x4 = df.iloc[:, 5].values  # 

y = df.iloc[:, 2].values  #

# Initialize parameters
m = len(y)  # number of training examples
x1 = x1.reshape(m, 1)
x2 = x2.reshape(m, 1)
x3 = x3.reshape(m, 1)


# Add a column of ones to x for the bias term
x = np.hstack((np.ones((m, 1)), x1,x2,x3))
y = y.reshape(m, 1)

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(x, y)


# The coefficients
print("Coefficients: \n", regr.coef_)



'''
#x4 = x4.reshape(m, 1)
alpha = 0.001
iterations = 10000

# Add a column of ones to x for the bias term
x = np.hstack((np.ones((m, 1)), x1,x2,x3))

theta = np.zeros((4, 1))  # theta parameters; it is a column vector


# Run batch gradient descent
for iteration in range(iterations):  # example number of iterations
    # define hypothesis that we want to be close to real values (y)
    h = np.dot(x, theta)  # matrix multiplication

    # batch gradient descent update rule
    theta -= alpha * 1/m * np.dot(x.T, (h - y))

print(theta)

'''
'''
# SGD
np.random.seed(43)  # for reproducibility
for iteration in range(iterations):
    shuffled_indices = np.random.permutation(m)
    x_shuffled = x[shuffled_indices]
    y_shuffled = y[shuffled_indices]
    for i in range(m):
        xi = x_shuffled[i:i+1]
        yi = y_shuffled[i:i+1]
        h = np.dot(xi, theta)
        gradient = np.dot(xi.T, (h - yi))
        theta -= alpha * gradient
'''
'''
print(theta)

# 05.01.2020 16:35:00.000 GMT-0600	1.11611	1.11628	1.11611	1.11626	27.39

#print(theta[0,0]*1 + theta[1,0]*1.11611 + theta[2,0]*1.11611  + theta[3,0]*1.11626 )


# Scatter plot of the data
plt.scatter(x[:, 1], y, color='red', marker='x', label='Training data')

# Line plot of the hypothesis
h = np.dot(x, theta)
plt.plot(x[:, 1], h, color='blue', label='Linear regression')

# Add labels
plt.xlabel('Open')
plt.ylabel('High')
plt.legend()

# Show the plot
plt.show()
'''
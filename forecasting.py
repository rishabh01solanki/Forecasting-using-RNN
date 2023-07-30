import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import make_interp_spline, BSpline

# Load data
df = pd.read_csv('/Users/rishabhsolanki/Desktop/Machine learning/one_day.csv')

# Select columns
X = df.iloc[:, 1:4].values
y = df.iloc[:, 2].values

# Normalize data
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Add ones column for bias
X = np.hstack((np.ones((X.shape[0], 1)), X))

# Split data into train and test sets
split_ratio = 0.8
split_idx = int(split_ratio * X.shape[0])
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Parameters
alpha = 0.01  # learning rate
iterations = 1000  # number of iterations

# Initialize theta
theta = np.zeros(X_train.shape[1])

# Gradient descent
def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    for it in range(iterations):
        prediction = np.dot(X, theta)
        theta = theta - (1/m) * alpha * (X.T.dot((prediction - y)))
    return theta

# Run gradient descent
theta = gradient_descent(X_train, y_train, theta, alpha, iterations)

# Create prediction
predictions = X_test.dot(theta)

# Initialize plot
fig, ax = plt.subplots(figsize=(10, 8))  # Change the values as per your preference
ln_actual, = plt.plot([], [], 'r-', animated=True, label='Actual')
ln_predicted, = plt.plot([], [], 'b-', animated=True, label='Predicted')

# Set plot labels
ax.set_xlabel('Day')
ax.set_ylabel('Price')
ax.set_title('Stock Price Forecasting using Logistic Regression')
ax.legend()

# For smoothness, interpolate data points for both actual and predicted
xnew = np.linspace(0, len(y_test), len(y_test) * 10)
spl_actual = make_interp_spline(range(len(y_test)), y_test, k=2)
spl_predicted = make_interp_spline(range(len(predictions)), predictions, k=2)
y_smooth_actual = spl_actual(xnew)
y_smooth_predicted = spl_predicted(xnew)

# Initialize animation
def init():
    ax.set_xlim(0, len(y_test))
    ax.set_ylim(1.1158, 1.1180)
    return ln_actual, ln_predicted,

# Update animation at each frame
def update(frame):
    ln_actual.set_data(xnew[:frame], y_smooth_actual[:frame])
    ln_predicted.set_data(xnew[:frame], y_smooth_predicted[:frame])
    return ln_actual, ln_predicted,

# Create animation
ani = FuncAnimation(fig, update, frames=len(xnew), init_func=init, blit=True, interval=50)  # Decreased interval for faster animation

# Show plot
plt.show()

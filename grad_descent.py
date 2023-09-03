import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv('/Users/rishabhsolanki/Desktop/Machine learning/Forecasting-using-RNN/synthetic_dataset.csv')

x = df.iloc[:, 1].values.astype(float)  # Size of houses
y = df.iloc[:, 0].values.astype(float)  # Price of houses

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Data normalization
x_scaler = StandardScaler()
y_scaler = StandardScaler()

x_train = x_scaler.fit_transform(x_train.reshape(-1, 1))
x_test = x_scaler.transform(x_test.reshape(-1, 1))
y_train = y_scaler.fit_transform(y_train.reshape(-1, 1))
y_test = y_scaler.transform(y_test.reshape(-1, 1))

# Define a linear regression model using TensorFlow
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=(1,))
])

# Compile the model
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)  # Experiment with different learning rates
model.compile(optimizer=optimizer, loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, epochs=100, verbose=0)

# Save the model
model.save('regression_model')

# Make predictions
y_pred = model.predict(x_test)

# Revert scaling for plotting
x_test_orig = x_scaler.inverse_transform(x_test)
y_pred_orig = y_scaler.inverse_transform(y_pred)
y_test_orig = y_scaler.inverse_transform(y_test)

# Plot the original data and the regression line
plt.scatter(x_test_orig, y_test_orig, color='red', label='Actual Data')
plt.plot(x_test_orig, y_pred_orig, color='blue', label='Regression Line')
plt.title('Size of houses vs Price (Linear Regression)')
plt.xlabel('Size of house')
plt.ylabel('Price')
plt.legend()
plt.show()

# Evaluate the model on the testing data
mse = tf.keras.losses.mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {np.mean(mse)}")

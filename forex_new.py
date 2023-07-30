import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv('/Users/rishabhsolanki/Desktop/Machine learning/one_day.csv')
X = df.iloc[:, 1:4].values
y = df.iloc[:, 2].values

# Standardize the features
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y.reshape(-1, 1))

# Split the data into train and test sets
split_ratio = 0.8
split_idx = int(split_ratio * X.shape[0])
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Train the SVM model
model = SVR(kernel='rbf', C=1e3, gamma=0.1)
model.fit(X_train, y_train.ravel())

# Forecast the future prices
predictions = model.predict(X_test)
predictions = predictions.reshape(-1, 1)  # Convert 1D array to 2D
predictions = sc_y.inverse_transform(predictions)  # Convert back to original scale

# Visualize the actual prices and predicted prices
plt.figure(figsize=(10, 8))
plt.plot(np.arange(y_train.shape[0], y_train.shape[0] + y_test.shape[0]), predictions.flatten(), color='blue', label='Predicted')
plt.plot(np.arange(0, y_train.shape[0] + y_test.shape[0]), sc_y.inverse_transform(np.vstack((y_train, y_test))).flatten(), color='red', label='Actual')
plt.xlabel('Day')
plt.ylabel('Price')
plt.title('Stock Price Forecasting using Support Vector Machine')
plt.legend()
plt.show()

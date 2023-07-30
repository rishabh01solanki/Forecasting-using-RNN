import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

# Number of bootstrap samples to create
n_bootstrap = 1000

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

# Matrix to hold bootstrap predictions
bootstrap_preds = np.zeros((n_bootstrap, len(X_test)))

# Generate bootstrap predictions
for i in range(n_bootstrap):
    X_train_sample, y_train_sample = resample(X_train, y_train.ravel())
    model = SVR(kernel='rbf', C=1e3, gamma=0.1)
    model.fit(X_train_sample, y_train_sample)
    predictions = model.predict(X_test)
    bootstrap_preds[i, :] = predictions

# Calculate prediction interval
lower_bound = np.percentile(bootstrap_preds, 2.5, axis=0)
upper_bound = np.percentile(bootstrap_preds, 97.5, axis=0)

# Convert to original scale
lower_bound = sc_y.inverse_transform(lower_bound.reshape(-1, 1))
upper_bound = sc_y.inverse_transform(upper_bound.reshape(-1, 1))
predictions = sc_y.inverse_transform(np.mean(bootstrap_preds, axis=0).reshape(-1, 1))

# Visualize the actual prices and predicted prices
plt.figure(figsize=(10, 8))
plt.plot(np.arange(y_train.shape[0], y_train.shape[0] + y_test.shape[0]), predictions.flatten(), color='blue', label='Predicted')
plt.fill_between(np.arange(y_train.shape[0], y_train.shape[0] + y_test.shape[0]), lower_bound.flatten(), upper_bound.flatten(), color='blue', alpha=0.2)
plt.plot(np.arange(0, y_train.shape[0] + y_test.shape[0]), sc_y.inverse_transform(np.vstack((y_train, y_test))).flatten(), color='red', label='Actual')
plt.xlabel('Day')
plt.ylabel('Price')
plt.title('Stock Price Forecasting using Support Vector Machine')
plt.legend()
plt.show()

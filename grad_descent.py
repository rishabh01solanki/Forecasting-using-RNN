import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import coremltools as ct

# Generate synthetic data
np.random.seed(0)
X = np.linspace(1000, 3000, 1000).reshape(-1, 1)
y = 0.3 * X + 50 + np.random.randn(1000, 1) * 50

# Scale the features and labels
scaler_x = StandardScaler()
scaler_y = StandardScaler()
X = scaler_x.fit_transform(X)
y = scaler_y.fit_transform(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(1,)),  # First hidden layer with 64 neurons and ReLU activation
    tf.keras.layers.Dense(32, activation='relu'),  # Second hidden layer with 32 neurons and ReLU activation
    tf.keras.layers.Dense(1)  # Output layer with 1 neuron (since we're doing regression)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=100)
#model.save('regression_model')
mlmodel = ct.convert(model, convert_to="mlprogram")
# Save the Core ML model
mlmodel.save("MyModel.mlpackage")


# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse_test = model.evaluate(X_test, y_test)
print(f'Test MSE: {mse_test}')

feature_mean = scaler_x.mean_[0]
feature_std = scaler_x.scale_[0]
label_mean = scaler_y.mean_[0]
label_std = scaler_y.scale_[0]

print("Feature Mean:", feature_mean)
print("Feature Std:", feature_std)
print("Label Mean:", label_mean)
print("Label Std:", label_std)


# Sort the test and predicted data for plotting
sorted_order = np.argsort(scaler_x.inverse_transform(X_test), axis=0).flatten()
sorted_x_test = scaler_x.inverse_transform(X_test)[sorted_order]
sorted_y_pred = scaler_y.inverse_transform(y_pred)[sorted_order]

# Plotting
plt.scatter(scaler_x.inverse_transform(X_test), scaler_y.inverse_transform(y_test), color='red', label='Actual')
plt.plot(sorted_x_test, sorted_y_pred, color='blue', label='Predicted Line')
plt.xlabel('Size of house (sq ft)')
plt.ylabel('Price in $1000')
plt.title('House Price Prediction')
plt.legend()
plt.show()

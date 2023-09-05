# Creating the content for the Jupyter Notebook (.ipynb) file
notebook_content = {
    "metadata": {
        "language_info": {
            "name": "python",
            "version": "3.9"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5,
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "# Forex Prediction using RNN"
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": """
import pandas as pd

# Load the data from the CSV file
# Update the file_path to point to the location of your CSV file
file_path = 'euro_us.csv'
df = pd.read_csv(file_path)

# Convert 'Local time' to datetime format
df['Local time'] = pd.to_datetime(df['Local time'].str.split(" ", expand=True)[0])

# Check for missing values
missing_values = df.isnull().sum()
missing_values
""",
            "execution_count": None,
            "outputs": []
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": """
from sklearn.preprocessing import MinMaxScaler

# Select feature columns
feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
features = df[feature_columns]

# Normalize the features
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)

# Convert scaled features to DataFrame
scaled_df = pd.DataFrame(scaled_features, columns=feature_columns)
scaled_df.head()
""",
            "execution_count": None,
            "outputs": []
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": """
import numpy as np

def create_sequences(data, seq_length):
    sequences = []
    next_values = []
    
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])
        next_values.append(data[i+seq_length, 3])
     
    return np.array(sequences), np.array(next_values)

# Define sequence length
seq_length = 60

# Create sequences
sequences, next_values = create_sequences(scaled_features, seq_length)
sequences.shape, next_values.shape
""",
            "execution_count": None,
            "outputs": []
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": """
import tensorflow as tf

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(seq_length, len(feature_columns))),
    tf.keras.layers.LSTM(50, return_sequences=True),
    tf.keras.layers.LSTM(50),
    tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Summary of the model
model.summary()
""",
            "execution_count": None,
            "outputs": []
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": """
# Split the data into training and validation sets
train_sequences = sequences[:200]
train_next_values = next_values[:200]
val_sequences = sequences[200:]
val_next_values = next_values[200:]

# Train the model
history = model.fit(
    train_sequences, train_next_values,
    epochs=7,
    batch_size=32,
    validation_data=(val_sequences, val_next_values)
)
""",
            "execution_count": None,
            "outputs": []
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": """
# Evaluate the model
loss = model.evaluate(val_sequences, val_next_values)
loss
""",
            "execution_count": None,
            "outputs": []
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": """
# Make predictions
predictions = model.predict(val_sequences)

# Reshape predictions and actual values
predictions = predictions.reshape(-1, 1)
val_next_values = val_next_values.reshape(-1, 1)

# Inverse transform using the new scaler
close_scaler = MinMaxScaler()
close_scaler.fit(df[['Close']])
inverse_predictions = close_scaler.inverse_transform(predictions)
inverse_actual = close_scaler.inverse_transform(val_next_values)
""",
            "execution_count": None,
            "outputs": []
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": """
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

# Make sure that the validation dates and predicted/actual values have the same length
val_length = len(val_next_values)

# Extract the corresponding dates for the validation set from the original DataFrame
val_dates = df['Local time'].tail(val_length).reset_index(drop=True)

# Plotting
plt.figure(figsize=(15, 6))
plt.plot(val_dates, inverse_actual, label='Actual')
plt.plot(val_dates, inverse_predictions, label='Predicted')

# Formatting the x-axis to display dates more clearly
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))
plt.gcf().autofmt_xdate()

plt.legend()
plt.title('Actual vs Predicted Close Prices')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.show()
""",
            "execution_count": None,
            "outputs": []
        }
    ]
}

# Save the notebook content to a .ipynb file
notebook_file_path = 'Forex_Prediction_RNN.ipynb'
import json

with open(notebook_file_path, 'w') as f:
    json.dump(notebook_content, f)

notebook_file_path

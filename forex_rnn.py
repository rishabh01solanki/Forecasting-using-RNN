import pandas as pd

# Load the data from the CSV file
file_path = '/Users/rishabhsolanki/Desktop/Machine learning/Forecasting-using-RNN/one_day.csv'
df = pd.read_csv(file_path)

# Display the first few rows of the dataframe
#print(df.head())

# Convert 'Local time' to datetime format
df['Local time'] = pd.to_datetime(df['Local time'].str.split(" ", expand=True)[0])

# Check for missing values
missing_values = df.isnull().sum()
#print(missing_values)

from sklearn.preprocessing import MinMaxScaler

# Select feature columns
feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
features = df[feature_columns]

# Normalize the features
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)

# Convert scaled features to DataFrame
scaled_df = pd.DataFrame(scaled_features, columns=feature_columns)

# Display first few rows of scaled data
#print(scaled_df.head())

import numpy as np
def create_sequences(data, seq_length):
    """
    Create sequences from the data.
    
    Parameters:
        data (np.array): 2D array of shape (num_samples, num_features)
        seq_length (int): Length of the sequence
    
    Returns:
        np.array: 3D array of shape (num_samples - seq_length, seq_length, num_features) containing sequences
        np.array: 1D array of shape (num_samples - seq_length,) containing the next value to predict
    """
    sequences = []
    next_values = []
    
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])
        next_values.append(data[i+seq_length, 3])  # Close price is at index 3
     
    return np.array(sequences), np.array(next_values)

# Define sequence length
seq_length = 60

# Create sequences
sequences, next_values = create_sequences(scaled_features, seq_length)

# Show shape of the sequences and next_values
#print(sequences.shape, next_values.shape)

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


# Split the data into training and validation sets
train_sequences = sequences[:200]
train_next_values = next_values[:200]
val_sequences = sequences[200:]
val_next_values = next_values[200:]


# Train the model
history = model.fit(
    train_sequences, train_next_values,
    epochs=10,
    batch_size=32,
    validation_data=(val_sequences, val_next_values)
)


# Evaluate the model
loss = model.evaluate(val_sequences, val_next_values)


# Make predictions
predictions = model.predict(val_sequences)

from sklearn.preprocessing import MinMaxScaler

# Assume 'predictions' is the array of scaled predictions
# and 'val_next_values' is the array of scaled actual values

close_scaler = MinMaxScaler()
close_scaler.fit(df[['Close']])

# Reshape predictions and actual values
predictions = predictions.reshape(-1, 1)
val_next_values = val_next_values.reshape(-1, 1)

# Inverse transform using the new scaler
inverse_predictions = close_scaler.inverse_transform(predictions)
inverse_actual = close_scaler.inverse_transform(val_next_values)

import matplotlib.dates as mdates
import matplotlib.pyplot as plt


# Assume 'val_dates' is a list of datetime objects corresponding to your validation set
# It should have the same length as 'inverse_actual' and 'inverse_predictions'


# Extract the corresponding dates for the validation set from the original DataFrame
val_dates = df['Local time'].tail(27).reset_index(drop=True)
plt.figure(figsize=(15, 6))

# Plotting the actual and predicted values
plt.plot(val_dates, inverse_actual, label='Actual')
plt.plot(val_dates, inverse_predictions, label='Predicted')

# Formatting the x-axis to display dates more clearly
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))
plt.gcf().autofmt_xdate()  # Rotation of x-axis labels for better visibility

plt.legend()
plt.title('Actual vs Predicted Close Prices')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.show()

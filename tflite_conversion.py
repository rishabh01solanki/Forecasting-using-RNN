# Import the required module
import tensorflow as tf

# Step 1: Load the saved model
model = tf.keras.models.load_model('linear_regression_model')

# Step 2: Convert the model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Step 3: Save the TFLite model
with open('linear_regression_model.tflite', 'wb') as f:
    f.write(tflite_model)

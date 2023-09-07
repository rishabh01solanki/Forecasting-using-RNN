import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


# Load Data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    'train_data',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',  # Change from 'binary' to 'categorical'
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    'train_data',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',  # Change from 'binary' to 'categorical'
    subset='validation'
)

# CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')  # softmax layer
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10
)

# CoreML Conversion
import coremltools as ct
from coremltools.models.neural_network import NeuralNetworkBuilder, SgdParams

# CoreML Conversion
input_shape = (1, 150, 150, 3)
input_shape_spec = ct.Shape(shape=input_shape)
input_spec = ct.ImageType(shape=input_shape_spec, bias=[0,0,0], scale=1/255.0)

coreml_model = ct.convert(model, inputs=[input_spec], source="tensorflow")

# Print layer names to figure out the correct name for the last dense layer
for layer in coreml_model.get_spec().neuralNetwork.layers:
    print(layer.name)

# ... (previous code remains unchanged)

# Create the NeuralNetworkBuilder with existing spec
builder = NeuralNetworkBuilder(spec=coreml_model.get_spec())

# Add a new softmax layer
builder.add_softmax(name='output_prob', input_name='sequential/dense_1/BiasAdd', output_name='output_prob')

# Identify layers to be made updatable (here, we are making the last dense layer updatable)
updatable_layers = ['sequential/dense_1/BiasAdd']
builder.make_updatable(updatable_layers)

# Set the number of epochs for on-device training
builder.set_epochs(10)

# Set the correct loss function and optimizer, use the output of the new softmax layer
builder.set_categorical_cross_entropy_loss(name="lossLayer", input="output_prob")
builder.set_sgd_optimizer(SgdParams(lr=0.01, batch=1))

# Compile and save the updated Core ML model
updatable_coreml_model = ct.models.MLModel(builder.spec)
updatable_coreml_model.save("cnn.mlmodel")

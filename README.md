import tensorflow as tf
from tensorflow.keras.datasets import boston_housing
from sklearn import preprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from sklearn.model_selection import train_test_split

# Load the Boston Housing dataset
(train_x, train_y), _ = boston_housing.load_data()

# Normalize the training data
train_x = preprocessing.normalize(train_x)

# Split the data into training and testing sets
train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size=0.2, random_state=42)

# Normalize the testing data
test_x = preprocessing.normalize(test_x)

# Build the model
model = Sequential([
    Dense(128, activation='relu', input_shape=(train_x[0].shape)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)
])

# Compile the model
model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

# Train the model using training data
history = model.fit(x=train_x, y=train_y, epochs=100, batch_size=1, verbose=1, validation_data=(test_x, test_y))

# Evaluate the model using testing data
evaluation_result = model.evaluate(test_x, test_y)
print("Evaluation Result:", evaluation_result)

# Test input for prediction
test_input = [[8.65407330e-05, 0.00000000e+00, 1.13392175e-02, 0.00000000e-00, 1.12518247e-03,
               1.31897603e-02, 7.53763011e-02, 1.30768051e-02, 1.09241016e-02, 4.89399752e-01,
               4.41333705e-02, 8.67155186e-01, 1.75004108e-02]]

# Print actual output and predicted output
print("Actual Output : 21.1", "\nPredicted output :", model.predict(test_input))

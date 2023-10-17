from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np

# Simulate some data
np.random.seed(0)  # for reproducibility
n_samples = 1000

# Cat data (height, weight)
cats = np.random.normal(25, 5, (n_samples, 2))  # cats generally weigh less and are smaller
# Dog data (height, weight)
dogs = np.random.normal(45, 10, (n_samples, 2))  # dogs generally weigh more and are bigger

# Create labels
labels = np.array([0]*n_samples + [1]*n_samples)  # 0 for cats, 1 for dogs

# Combine the data
data = np.vstack((cats, dogs))

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=1)

# Define the model
model = Sequential([
    Dense(10, activation='relu', input_shape=(2,)),  # Hidden layer with 10 neurons
    Dense(1, activation='sigmoid')  # Output layer
])

# Compile the model
model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=20, batch_size=32)

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)

print("Test Accuracy: ", accuracy)

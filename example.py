import tensorflow as tf
from tensorflow.keras import layers

# Load MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize images to [0,1]
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Add channel dimension: (28,28) -> (28,28,1)
x_train = x_train[..., None]
x_test = x_test[..., None]

# CNN model
model = tf.keras.Sequential([
    layers.Conv2D(8, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(10, activation="softmax")
])

# Configure training
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train
model.fit(x_train, y_train, epochs=5)

# Test
loss, accuracy = model.evaluate(x_test, y_test)
print("Test accuracy:", accuracy)
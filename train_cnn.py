import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
import numpy as np
# Dataset path
DATASET_DIR = Path("dataset")
# Image settings
IMG_SIZE = (64, 64)
BATCH_SIZE = 32
SEED = 42
# Load training dataset
train_ds = keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    labels="inferred",
    label_mode="binary",
    class_names=["closed", "open"],
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale"
)
# Load validation dataset
val_ds = keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    labels="inferred",
    label_mode="binary",
    class_names=["closed", "open"],
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale"
)
# Normalize pixel values: 0-255 -> 0-1
normalization = layers.Rescaling(1.0 / 255)
train_ds = train_ds.map(
    lambda images, labels: (normalization(images), labels)
)
val_ds = val_ds.map(
    lambda images, labels: (normalization(images), labels)
)
# Improve input pipeline performance
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
# CNN model
model = keras.Sequential([
    layers.Input(shape=(64, 64, 1)),
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(1, activation="sigmoid")
])
# Configure model training
# Calculate class weights
class_weights = {
    0: 2.10,  # Closed eyes
    1: 0.66   # Open eyes
}
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Print architecture
model.summary()
# Train the CNN model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    class_weight=class_weights
)

# Save the trained model
model.save("eye_state_cnn.keras")
print("Model saved successfully!")

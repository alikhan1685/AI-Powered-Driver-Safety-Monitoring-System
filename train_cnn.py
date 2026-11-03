import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path

# Dataset path
DATASET_DIR = Path("dataset")

# Image settings
IMG_SIZE = (64, 64)
BATCH_SIZE = 32
SEED = 42

# 1. Load training dataset
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

# 2. Load validation dataset
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

# 3. Data Augmentation (Naya Add Kiya Hai)
# Ye model ko webcam ki alag lighting aur angles ke liye tayyar karega
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),       # Thora rotate
    layers.RandomZoom(0.05),           # Thora zoom in/out
    layers.RandomBrightness(0.15),     # Lighting ki aadat
    layers.RandomContrast(0.15)        # Contrast ki aadat
])

# 4. Normalize pixel values: 0-255 -> 0-1
normalization = layers.Rescaling(1.0 / 255)

# Augmentation sirf training par lagayein, validation par nahi
train_ds = train_ds.map(
    lambda images, labels: (data_augmentation(images, training=True), labels)
)
train_ds = train_ds.map(
    lambda images, labels: (normalization(images), labels)
)
val_ds = val_ds.map(
    lambda images, labels: (normalization(images), labels)
)

# Improve input pipeline performance
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

# 5. Improved CNN Model (Thora Deep)
model = keras.Sequential([
    layers.Input(shape=(64, 64, 1)),
    
    # Layer 1
    layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    # Layer 2
    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    # Layer 3 (Naya Add Kiya)
    layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.4), # Overfitting roknay ke liye
    layers.Dense(1, activation="sigmoid")
])

# 6. Configure model training
class_weights = {
    0: 2.10,  # Closed eyes
    1: 0.66   # Open eyes
}

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
model.summary()
# 7. Train the CNN model
# Callbacks: Best model save karne ke liye aur early stopping
callbacks = [
    keras.callbacks.ModelCheckpoint(
        "eye_state_cnn.keras", 
        save_best_only=True, 
        monitor="val_accuracy"
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_loss", 
        patience=3, 
        restore_best_weights=True
    )
]
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15, # 15 epochs dein
    class_weight=class_weights,
    callbacks=callbacks
)
print("Model saved successfully!")
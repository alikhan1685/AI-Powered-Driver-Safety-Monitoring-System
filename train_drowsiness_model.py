import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path

# --- CONFIGURATION ---
BASE_DIR = Path(r"D:\AI-Powered Truck Driver Safety Monitoring System\archive (1)\data")
TRAIN_DIR = BASE_DIR / "train"
VAL_DIR   = BASE_DIR / "val"

IMG_SIZE = (96, 96)      # Face crops (larger than eye crops)
BATCH_SIZE = 32
SEED = 42

# --- LOAD DATA ---
train_ds = keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="binary",
    class_names=["awake", "sleepy"],   # 0 = awake, 1 = sleepy
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="rgb"
)

val_ds = keras.utils.image_dataset_from_directory(
    VAL_DIR,
    labels="inferred",
    label_mode="binary",
    class_names=["awake", "sleepy"],
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="rgb"
)

print("Class indices:", train_ds.class_names)

# --- AUGMENTATION ---
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.05),
    layers.RandomBrightness(0.15),
    layers.RandomContrast(0.15)
])

normalization = layers.Rescaling(1.0 / 255)

train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))
train_ds = train_ds.map(lambda x, y: (normalization(x), y))
val_ds   = val_ds.map(lambda x, y: (normalization(x), y))

train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds   = val_ds.prefetch(tf.data.AUTOTUNE)

# --- MODEL ---
model = keras.Sequential([
    layers.Input(shape=(96, 96, 3)),

    layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.4),
    layers.Dense(1, activation="sigmoid")   # 0 = awake, 1 = sleepy
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# --- CALLBACKS ---
callbacks = [
    keras.callbacks.ModelCheckpoint(
        "drowsiness_cnn.keras",
        save_best_only=True,
        monitor="val_accuracy"
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )
]

# --- TRAIN ---
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    callbacks=callbacks
)

print("Drowsiness model saved as drowsiness_cnn.keras")
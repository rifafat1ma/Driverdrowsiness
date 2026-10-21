import os
import tensorflow as tf
from tensorflow.keras import layers, models

DATASET_DIR = "dataset_eye"
MODEL_DIR = "models"

IMG_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 10
SEED = 42

os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------------
# LOAD DATA
# -------------------------
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE
)

print("CLASS NAMES:", train_ds.class_names)

# Normalize 0-255 -> 0-1
normalization = layers.Rescaling(1.0 / 255)

# -------------------------
# SMALL CNN
# -------------------------
model = models.Sequential([
    layers.Input(shape=(64, 64, 1)),

    normalization,

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Flatten(),

    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),

    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "models/eye_small_best.keras",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stop]
)

model.save("models/eye_small.keras")

print("\nTraining finished.")
print("Best model: models/eye_small_best.keras")
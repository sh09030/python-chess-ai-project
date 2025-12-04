import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print("TF Version:", tf.__version__)

# Load data
X = np.load("x_positions.npy")
y = np.load("y_evals.npy")

# Normalize
max_cp = 2000
y_norm = np.clip(y, -max_cp, max_cp) / max_cp

# Build model
model = keras.Sequential([
    layers.Input(shape=(8, 8, 12)),
    layers.Conv2D(32, (3,3), padding="same", activation="relu"),
    layers.Conv2D(64, (3,3), padding="same", activation="relu"),
    layers.Conv2D(64, (3,3), padding="same", activation="relu"),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(1, activation="tanh")
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0005),
    loss="mse",
    metrics=["mae"]
)

# Train
history = model.fit(
    X, y_norm,
    batch_size=128,
    epochs=10,
    validation_split=0.1
)

# Save
model.save("eval_net.h5")
print("Saved eval_net.h5")

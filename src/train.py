import csv
import os

import numpy as np
import tensorflow as tf
import yaml
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout

PROCESSED_DIR = "data/processed"
MODEL_DIR = "models"

with open("params.yaml", "r") as file:
    params = yaml.safe_load(file)

train_params = params["train"]

dense_units = train_params["dense_units"]
dropout_rate = train_params["dropout_rate"]
learning_rate = train_params["learning_rate"]
epochs = train_params["epochs"]
batch_size = train_params["batch_size"]

os.makedirs(MODEL_DIR, exist_ok=True)

train_data = np.load(os.path.join(PROCESSED_DIR, "train.npz"))
val_data = np.load(os.path.join(PROCESSED_DIR, "val.npz"))

x_train = train_data["images"]
y_train = train_data["labels"]
x_val = val_data["images"]
y_val = val_data["labels"]

model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(dense_units, activation="relu"),
    Dropout(dropout_rate),
    Dense(10, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=epochs,
    batch_size=batch_size
)

model.save(os.path.join(MODEL_DIR, "model.h5"))

with open(os.path.join(MODEL_DIR, "history.csv"), "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["epoch", "loss", "accuracy", "val_loss", "val_accuracy"])

    for epoch in range(len(history.history["loss"])):
        writer.writerow([
            epoch + 1,
            history.history["loss"][epoch],
            history.history["accuracy"][epoch],
            history.history["val_loss"][epoch],
            history.history["val_accuracy"][epoch]
        ])

print("Model saved to models/model.h5")
print("Training history saved to models/history.csv")

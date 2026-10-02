import os
import numpy as np
from tensorflow.keras.datasets import fashion_mnist

RAW_DIR = "data/raw"

os.makedirs(RAW_DIR, exist_ok=True)

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

np.savez_compressed(
    os.path.join(RAW_DIR, "train.npz"),
    images=x_train,
    labels=y_train
)

np.savez_compressed(
    os.path.join(RAW_DIR, "test.npz"),
    images=x_test,
    labels=y_test
)

print("Fashion-MNIST raw data saved to data/raw/")

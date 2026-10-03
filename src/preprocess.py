import os
import yaml
import numpy as np
from sklearn.model_selection import train_test_split

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

with open("params.yaml", "r") as file:
    params = yaml.safe_load(file)

preprocess_params = params["preprocess"]
test_size = preprocess_params["test_size"]
seed = preprocess_params["seed"]

os.makedirs(PROCESSED_DIR, exist_ok=True)

train_data = np.load(os.path.join(RAW_DIR, "train.npz"))
test_data = np.load(os.path.join(RAW_DIR, "test.npz"))

# Normalize pixel values to [0, 1]
x_train = train_data["images"].astype("float32") / 255.0
y_train = train_data["labels"]

x_test = test_data["images"].astype("float32") / 255.0
y_test = test_data["labels"]

# Explicitly clip normalized values to [0, 1]
x_train = np.clip(x_train, 0.0, 1.0)
x_test = np.clip(x_test, 0.0, 1.0)

x_train, x_val, y_train, y_val = train_test_split(
    x_train,
    y_train,
    test_size=test_size,
    random_state=seed,
    stratify=y_train
)

# Clip validation data as well
x_val = np.clip(x_val, 0.0, 1.0)

np.savez_compressed(
    os.path.join(PROCESSED_DIR, "train.npz"),
    images=x_train,
    labels=y_train
)

np.savez_compressed(
    os.path.join(PROCESSED_DIR, "val.npz"),
    images=x_val,
    labels=y_val
)

np.savez_compressed(
    os.path.join(PROCESSED_DIR, "test.npz"),
    images=x_test,
    labels=y_test
)

print("Processed train, validation, and test data saved to data/processed/")
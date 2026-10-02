import json
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

MODEL_PATH = "models/model.h5"
TEST_PATH = "data/processed/test.npz"

data = np.load(TEST_PATH)

x_test = data["images"]
y_test = data["labels"]

model = tf.keras.models.load_model(MODEL_PATH)

test_loss, test_accuracy = model.evaluate(
    x_test,
    y_test,
    verbose=0
)

predictions = model.predict(x_test, verbose=0)
y_pred = np.argmax(predictions, axis=1)

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot"
    ]
)

disp.plot(xticks_rotation="vertical")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()

metrics = {
    "test_loss": float(test_loss),
    "test_accuracy": float(test_accuracy)
}

with open("metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")
print("Confusion matrix saved to confusion_matrix.png")
print("Metrics saved to metrics.json")

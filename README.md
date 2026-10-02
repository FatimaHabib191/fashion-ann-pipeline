# Fashion-MNIST ANN Pipeline with Git & DVC

An end-to-end machine learning pipeline for **Fashion-MNIST image classification** using a fully-connected TensorFlow Artificial Neural Network (ANN), with **Git for source-code versioning** and **DVC with Google Drive for data and model versioning**.

## Overview

This project implements a reproducible Fashion-MNIST classification workflow covering:

* Advanced Git version control and branching workflows
* A modular TensorFlow ANN pipeline
* DVC-based versioning of datasets and trained models
* Google Drive as the DVC remote
* Parameterized ML experiments using `params.yaml`
* Reproducible pipeline execution using `dvc repro`
* Git and DVC conflict resolution

The target is **at least 85% test accuracy** on the Fashion-MNIST test set.

## Dataset

The project uses the **Fashion-MNIST** dataset containing:

* 70,000 grayscale images
* Image size: 28 × 28 pixels
* 10 clothing categories
* 60,000 training images
* 10,000 test images

The dataset is loaded directly through TensorFlow/Keras.

## Project Structure

```text
fashion-ann-pipeline/
│
├── src/
│   ├── prepare.py
│   ├── preprocess.py
│   ├── train.py
│   └── evaluate.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── params.yaml
├── dvc.yaml
├── dvc.lock
├── metrics.json
├── .gitignore
└── README.md
```

## Machine Learning Pipeline

The ML pipeline consists of four independent stages.

### 1. Data Preparation

`src/prepare.py`

* Loads Fashion-MNIST using `tf.keras.datasets.fashion_mnist`
* Saves the raw images and labels under `data/raw/`

### 2. Data Preprocessing

`src/preprocess.py`

* Loads the raw dataset
* Normalizes pixel values to `[0, 1]`
* Creates a validation split from the training data
* Saves the processed train, validation, and test data under `data/processed/`

### 3. Model Training

`src/train.py`

Builds and trains a fully-connected ANN with the architecture:

```text
Flatten
   ↓
Dense (ReLU)
   ↓
Dropout
   ↓
Dense (10, Softmax)
```

The model uses:

* Adam optimizer
* Sparse categorical cross-entropy
* Hyperparameters loaded from `params.yaml`

The trained model is saved as:

```text
models/model.h5
```

Training history is saved as:

```text
models/history.csv
```

### 4. Model Evaluation

`src/evaluate.py`

* Loads the trained model
* Evaluates it on the processed test dataset
* Calculates test loss and accuracy
* Generates a confusion matrix
* Saves evaluation results to `metrics.json`

## Configuration

`params.yaml` provides the central configuration for the ML pipeline.

It contains parameters such as:

```text
preprocess:
  test_size
  seed

train:
  dense_units
  dropout_rate
  learning_rate
  epochs
  batch_size
```

The training and preprocessing scripts read these values instead of hard-coding hyperparameters.

## DVC

DVC is used to version the large ML artifacts:

```text
data/raw/
data/processed/
models/
```

The artifacts are tracked through DVC pointer files while the actual data and model files are stored in a **Google Drive DVC remote**.

The DVC pipeline is defined in:

```text
dvc.yaml
```

and the reproducible pipeline state is recorded in:

```text
dvc.lock
```

## Reproducible Pipeline

The complete pipeline contains four DVC stages:

```text
prepare
   ↓
preprocess
   ↓
train
   ↓
evaluate
```

The complete workflow can be reproduced with:

```bash
dvc repro
```

Changing a parameter in `params.yaml` causes DVC to identify and re-run the affected pipeline stages.

## Git Workflow

The project demonstrates the required Git operations, including:

* Repository initialization
* Feature development using the `dev` branch
* Multiple incremental commits
* Git log variants
* Git diff variants
* Stashing and restoring changes
* Rebasing `dev` onto `main`
* Soft and hard reset
* File organization using `git mv`
* File removal using `git rm`

The complete commit history is intentionally preserved and is not squashed.

## Conflict Resolution

The project also simulates collaboration between two contributors.

Two branches independently modify overlapping preprocessing code and DVC-tracked processed data. The resulting conflicts include:

* A Git code conflict in `preprocess.py`
* A DVC pointer conflict for the processed dataset

The conflicts are resolved by reconciling the preprocessing code and selecting/regenerating the appropriate DVC data version. The final state is verified using:

```bash
dvc status
dvc repro
```

## Technologies Used

* **Python 3.9+**
* **TensorFlow / Keras**
* **Scikit-learn**
* **PyYAML**
* **Matplotlib**
* **Git & GitHub**
* **DVC**
* **Google Drive**

## Running the Pipeline

Install the required dependencies:

```bash
pip install tensorflow dvc "dvc[gdrive]" pyyaml scikit-learn matplotlib
```

Run the complete reproducible pipeline:

```bash
dvc repro
```

To push DVC-tracked artifacts to the configured Google Drive remote:

```bash
dvc push
```

## Assignment Outcome

The completed project demonstrates an end-to-end, version-controlled and reproducible machine learning workflow:

```text
Fashion-MNIST
      ↓
Data Preparation
      ↓
Preprocessing
      ↓
ANN Training
      ↓
Evaluation
      ↓
Metrics
```

Git manages the source code and project history, while DVC manages datasets, models, and pipeline reproducibility.

## Author

**Assignment 3 — End-to-End ML Versioning with Git, DVC & Google Drive**

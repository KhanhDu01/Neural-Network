"""
Train.py (all-classes-at-once version)

Prereq: run build_cache.py first so data/cache/<category>.npy exists for every
category in data/categories.txt.

Usage:
    python Train.py             # fresh model
    python Train.py continue    # resume from main/model.json
"""
import sys
import json
import time
from pathlib import Path

import numpy as np

from Layer import (
    Layer_Dense,
    Activation_ReLU,
    Activation_Softmax_Loss_Categoricalentropy,
)

DATA_DIR = Path("data")
CACHE_DIR = DATA_DIR / "cache"
MODEL_PATH = Path("main/model.json")

EPOCHS = 10
BATCH_SIZE = 64
VAL_FRACTION = 0.05


class Model:
    def __init__(self, dense1, activation1, dense2, activation2, dense3, loss_activation):
        self.dense1 = dense1
        self.activation1 = activation1
        self.dense2 = dense2
        self.activation2 = activation2
        self.dense3 = dense3
        self.loss_activation = loss_activation

    @property
    def denses(self):
        return [self.dense1, self.dense2, self.dense3]


def initialize_or_load_model(n_categories, initialize=False):
    if initialize:
        dense1 = Layer_Dense(784, 128)
        dense2 = Layer_Dense(128, 128)
        dense3 = Layer_Dense(128, n_categories)
    else:
        with open(MODEL_PATH, "r") as f:
            saved = json.load(f)

        denses = []
        for i in (1, 2, 3):
            d = Layer_Dense(saved[f"n_weights_layer{i}"], saved[f"n_biases_layer{i}"])
            d.weights = np.array(saved[f"weights_layer{i}"])
            d.biases = np.array(saved[f"biases_layer{i}"])
            d.opt_weights.m = np.array(saved[f"opt_weights_m_layer{i}"])
            d.opt_weights.v = np.array(saved[f"opt_weights_v_layer{i}"])
            d.opt_weights.t = int(saved[f"opt_weights_t_layer{i}"])
            d.opt_biases.m = np.array(saved[f"opt_biases_m_layer{i}"])
            d.opt_biases.v = np.array(saved[f"opt_biases_v_layer{i}"])
            d.opt_biases.t = int(saved[f"opt_biases_t_layer{i}"])
            denses.append(d)
        dense1, dense2, dense3 = denses

    return Model(
        dense1, Activation_ReLU(),
        dense2, Activation_ReLU(),
        dense3, Activation_Softmax_Loss_Categoricalentropy(),
    )


def save_model(model, categories):
    params = {}
    for i, d in enumerate(model.denses, start=1):
        params[f"n_weights_layer{i}"] = d.weights.shape[0]
        params[f"weights_layer{i}"] = d.weights.tolist()
        params[f"opt_weights_m_layer{i}"] = d.opt_weights.m.tolist()
        params[f"opt_weights_v_layer{i}"] = d.opt_weights.v.tolist()
        params[f"opt_weights_t_layer{i}"] = int(d.opt_weights.t)
        params[f"n_biases_layer{i}"] = d.weights.shape[1]
        params[f"biases_layer{i}"] = d.biases.tolist()
        params[f"opt_biases_m_layer{i}"] = d.opt_biases.m.tolist()
        params[f"opt_biases_v_layer{i}"] = d.opt_biases.v.tolist()
        params[f"opt_biases_t_layer{i}"] = int(d.opt_biases.t)
    params["categories"] = categories

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "w") as f:
        json.dump(params, f)


def load_categories():
    with open(DATA_DIR / "categories.txt", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_all(categories):
    Xs, ys = [], []
    for i, c in enumerate(categories):
        arr = np.load(CACHE_DIR / f"{c}.npy")
        Xs.append(arr)
        ys.append(np.full(len(arr), i, dtype=np.int64))
    return np.concatenate(Xs), np.concatenate(ys)


def forward(model, X_batch):
    model.dense1.forward(X_batch)
    model.activation1.forward(model.dense1.output)
    model.dense2.forward(model.activation1.output)
    model.activation2.forward(model.dense2.output)
    model.dense3.forward(model.activation2.output)


def evaluate(model, X, y, n_classes, batch_size=512):
    correct = 0
    total_loss = 0.0
    for s in range(0, len(X), batch_size):
        X_batch = X[s:s + batch_size].astype(np.float32) / 255.0
        y_batch = np.eye(n_classes, dtype=np.float32)[y[s:s + batch_size]]
        forward(model, X_batch)
        loss = model.loss_activation.forward(model.dense3.output, y_batch)
        total_loss += loss * len(X_batch)
        correct += np.sum(np.argmax(model.loss_activation.output, axis=1) == y[s:s + batch_size])
    return total_loss / len(X), correct / len(X)


def train(model, X, y, X_val, y_val, categories, epochs=EPOCHS, batch_size=BATCH_SIZE):
    n_classes = len(categories)

    for epoch in range(epochs):
        t0 = time.time()
        idx = np.random.permutation(len(X))
        total_loss, correct = 0.0, 0

        for s in range(0, len(X), batch_size):
            b = idx[s:s + batch_size]
            X_batch = X[b].astype(np.float32) / 255.0
            y_batch = np.eye(n_classes, dtype=np.float32)[y[b]]

            # Forward
            forward(model, X_batch)
            loss = model.loss_activation.forward(model.dense3.output, y_batch)

            # Backward
            model.loss_activation.backward(model.loss_activation.output, y_batch)
            model.dense3.backward(model.loss_activation.dinputs)
            model.activation2.backward(model.dense3.dinputs)
            model.dense2.backward(model.activation2.dinputs)
            model.activation1.backward(model.dense2.dinputs)
            model.dense1.backward(model.activation1.dinputs)

            # Update
            model.dense1.update_params()
            model.dense2.update_params()
            model.dense3.update_params()

            total_loss += loss * len(b)
            correct += np.sum(np.argmax(model.loss_activation.output, axis=1) == y[b])

        val_loss, val_acc = evaluate(model, X_val, y_val, n_classes)
        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Loss: {total_loss / len(X):.4f} | Acc: {correct / len(X):.2%} | "
            f"Val loss: {val_loss:.4f} | Val acc: {val_acc:.2%} | "
            f"Time: {time.time() - t0:.1f}s"
        )
        save_model(model, categories)  # checkpoint every epoch


def main():
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "initialize"
    if mode not in ("initialize", "continue"):
        print("Usage: python Train.py [continue]")
        return

    categories = load_categories()
    if not categories:
        raise ValueError("No categories found in data/categories.txt")

    print("Loading cached data...")
    X, y = load_all(categories)
    print(f"X: {X.shape}, y: {y.shape}")

    # Fixed-seed split so the validation set is identical across resumes
    rng = np.random.RandomState(0)
    perm = rng.permutation(len(X))
    n_val = int(len(X) * VAL_FRACTION)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    X_val, y_val = X[val_idx], y[val_idx]
    X_train, y_train = X[train_idx], y[train_idx]

    if mode == "initialize":
        print("Initializing a new model...")
        model = initialize_or_load_model(len(categories), initialize=True)
    else:
        with open(MODEL_PATH, "r") as f:
            if json.load(f).get("categories") != categories:
                raise ValueError("Saved model's categories don't match categories.txt")
        print("Resuming from saved model...")
        model = initialize_or_load_model(len(categories), initialize=False)

    try:
        train(model, X_train, y_train, X_val, y_val, categories)
    except KeyboardInterrupt:
        print("\nInterrupted. Last completed epoch is saved; run 'python Train.py continue' to resume.")


if __name__ == "__main__":
    main()
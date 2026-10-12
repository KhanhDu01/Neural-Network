"""
TrainCNN.py - convolutional version of the QuickDraw classifier.

Architecture (input is the same cached 28x28 uint8 images the MLP used):
    Conv3x3(1->C1) -> ReLU -> MaxPool2
    Conv3x3(C1->C2) -> ReLU -> MaxPool2
    Flatten -> Dense(HIDDEN) -> ReLU -> Dropout -> Dense(345)

Prereq: data/cache/<category>.npy for every category in data/categories.txt
        (same cache build_cache.py made for the MLP).

Usage:
    python TrainCNN.py             # fresh model
    python TrainCNN.py continue    # resume from main/model_cnn.npz
"""
import os
import sys
import time
from pathlib import Path

import numpy as np

from Layer import (
    Layer_Dense,
    Activation_ReLU,
    Activation_Softmax_Loss_Categoricalentropy,
)
from CNN import (
    Layer_Conv2D,
    Layer_MaxPool2D,
    Layer_Flatten,
    Layer_Dropout,
    make_adam,
)

DATA_DIR = Path("data")
CACHE_DIR = DATA_DIR / "cache"
MODEL_PATH = Path("main/model_cnn.npz")

# ---- Architecture ----
CONV1_CHANNELS = 32          # 32/64 is more accurate but ~2x slower per step
CONV2_CHANNELS = 64
HIDDEN = 256
DROPOUT = 0.15

# ---- Training ----
LEARNING_RATE = 0.001
LR_DECAY = 0.96              # lr is multiplied by this after every epoch
BATCH_SIZE = 128
EPOCHS = 20
SAMPLES_PER_EPOCH = 500_000  # one "epoch" = this many random training samples
VAL_FRACTION = 0.05          # same split as the MLP script, so numbers are comparable
VAL_SAMPLES = 50_000         # validation subset used after each epoch (speed)
PATIENCE = 4                 # stop after this many epochs without val-loss improvement


def make_dense(n_in, n_out, lr):
    """Layer_Dense from Layer.py, but float32 + He init + float32 Adam state."""
    d = Layer_Dense(n_in, n_out)
    d.weights = (np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)).astype(np.float32)
    d.biases = np.zeros((1, n_out), dtype=np.float32)
    d.opt_weights = make_adam(d.weights.shape, lr)
    d.opt_biases = make_adam(d.biases.shape, lr)
    return d


class CNN:
    def __init__(self, n_classes, c1=CONV1_CHANNELS, c2=CONV2_CHANNELS,
                 hidden=HIDDEN, dropout=DROPOUT, lr=LEARNING_RATE):
        self.n_classes, self.c1, self.c2, self.hidden = n_classes, c1, c2, hidden
        self.dropout_rate = dropout
        self.lr = lr
        self.best_val = float("inf")

        self.conv1 = Layer_Conv2D(1, c1, learning_rate=lr)
        self.conv1.need_input_grad = False  # nothing upstream of the image
        self.relu1 = Activation_ReLU()
        self.pool1 = Layer_MaxPool2D()

        self.conv2 = Layer_Conv2D(c1, c2, learning_rate=lr)
        self.relu2 = Activation_ReLU()
        self.pool2 = Layer_MaxPool2D()

        self.flatten = Layer_Flatten()
        self.dense1 = make_dense(c2 * 7 * 7, hidden, lr)
        self.relu3 = Activation_ReLU()
        self.dropout = Layer_Dropout(dropout)
        self.dense2 = make_dense(hidden, n_classes, lr)

        self.loss_activation = Activation_Softmax_Loss_Categoricalentropy()

        self.layers = [
            self.conv1, self.relu1, self.pool1,
            self.conv2, self.relu2, self.pool2,
            self.flatten, self.dense1, self.relu3, self.dropout, self.dense2,
        ]
        self.trainable = {
            "conv1": self.conv1, "conv2": self.conv2,
            "dense1": self.dense1, "dense2": self.dense2,
        }

    # ---------------- forward / backward ----------------
    def forward(self, X, training=True):
        """X: float32 (N, 784) in [0, 1]. Returns logits."""
        x = X.reshape(-1, 1, 28, 28)
        for layer in self.layers:
            if layer is self.dropout:
                layer.forward(x, training)
            else:
                layer.forward(x)
            x = layer.output
        return x

    def train_step(self, X, y):
        """One optimizer step on a batch. y: int class indices. Returns (loss, probs)."""
        logits = self.forward(X, training=True)
        loss = self.loss_activation.forward(logits, y)
        probs = self.loss_activation.output

        self.loss_activation.backward(probs, y)
        grad = self.loss_activation.dinputs
        for layer in reversed(self.layers):
            layer.backward(grad)
            grad = layer.dinputs

        for layer in self.trainable.values():
            layer.update_params()
        return loss, probs

    def set_lr(self, lr):
        self.lr = lr
        for layer in self.trainable.values():
            layer.opt_weights.lr = lr
            layer.opt_biases.lr = lr

    def predict_proba(self, X, batch_size=512):
        """X: (N, 784) or (784,), uint8 0-255 or float 0-1. Returns (N, n_classes) probabilities."""
        X = np.asarray(X)
        if X.ndim == 1:
            X = X[None, :]
        out = []
        for s in range(0, len(X), batch_size):
            xb = X[s:s + batch_size]
            xb = xb.astype(np.float32) / 255.0 if xb.dtype == np.uint8 else xb.astype(np.float32, copy=False)
            logits = self.forward(xb, training=False)
            e = np.exp(logits - logits.max(axis=1, keepdims=True))
            out.append(e / e.sum(axis=1, keepdims=True))
        return np.concatenate(out)

    # ---------------- save / load ----------------
    def save(self, path, categories):
        d = {
            "config": np.array([self.c1, self.c2, self.hidden, self.n_classes], dtype=np.int64),
            "dropout": np.float32(self.dropout_rate),
            "lr": np.float64(self.lr),
            "best_val": np.float64(self.best_val),
            "categories": np.array(categories),
        }
        for name, layer in self.trainable.items():
            d[f"{name}_w"] = layer.weights
            d[f"{name}_wm"] = layer.opt_weights.m
            d[f"{name}_wv"] = layer.opt_weights.v
            d[f"{name}_wt"] = np.int64(layer.opt_weights.t)
            d[f"{name}_b"] = layer.biases
            d[f"{name}_bm"] = layer.opt_biases.m
            d[f"{name}_bv"] = layer.opt_biases.v
            d[f"{name}_bt"] = np.int64(layer.opt_biases.t)

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.stem + ".tmp.npz")
        np.savez(tmp, **d)
        os.replace(tmp, path)  # atomic: a crash mid-save can't corrupt the checkpoint

    @classmethod
    def load(cls, path=MODEL_PATH):
        z = np.load(path, allow_pickle=False)
        c1, c2, hidden, n_classes = [int(v) for v in z["config"]]
        model = cls(n_classes, c1, c2, hidden, float(z["dropout"]), float(z["lr"]))
        model.best_val = float(z["best_val"])
        for name, layer in model.trainable.items():
            layer.weights = z[f"{name}_w"].astype(np.float32)
            layer.biases = z[f"{name}_b"].astype(np.float32)
            layer.opt_weights.m = z[f"{name}_wm"].astype(np.float32)
            layer.opt_weights.v = z[f"{name}_wv"].astype(np.float32)
            layer.opt_weights.t = int(z[f"{name}_wt"])
            layer.opt_biases.m = z[f"{name}_bm"].astype(np.float32)
            layer.opt_biases.v = z[f"{name}_bv"].astype(np.float32)
            layer.opt_biases.t = int(z[f"{name}_bt"])
        categories = [str(c) for c in z["categories"]]
        return model, categories


# ---------------- data ----------------
def load_categories():
    with open(DATA_DIR / "categories.txt", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_all(categories):
    Xs, ys = [], []
    for i, c in enumerate(categories):
        path = CACHE_DIR / f"{c}.npy"
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing - run build_cache.py first")
        arr = np.load(path)
        Xs.append(arr)
        ys.append(np.full(len(arr), i, dtype=np.int64))
    return np.concatenate(Xs), np.concatenate(ys)


def evaluate(model, X, y, idx, batch_size=512):
    """Returns (loss, top-1 accuracy, top-5 accuracy) over X[idx]."""
    n = len(idx)
    loss_sum, top1, top5 = 0.0, 0, 0
    for s in range(0, n, batch_size):
        b = idx[s:s + batch_size]
        xb = X[b].astype(np.float32) / 255.0
        yb = y[b]
        logits = model.forward(xb, training=False)
        loss = model.loss_activation.forward(logits, yb)
        probs = model.loss_activation.output

        loss_sum += loss * len(b)
        top1 += np.sum(probs.argmax(axis=1) == yb)
        best5 = np.argpartition(probs, -5, axis=1)[:, -5:]
        top5 += np.sum((best5 == yb[:, None]).any(axis=1))
    return loss_sum / n, top1 / n, top5 / n


# ---------------- training ----------------
def train(model, X, y, train_idx, val_idx, categories, epochs=EPOCHS,
          batch_size=BATCH_SIZE, samples_per_epoch=SAMPLES_PER_EPOCH,
          patience=PATIENCE, model_path=MODEL_PATH, verbose=True):
    bad = 0
    for epoch in range(epochs):
        t0 = time.time()
        n = min(samples_per_epoch, len(train_idx))
        order = np.random.permutation(train_idx)[:n]
        total_loss, correct = 0.0, 0

        for step, s in enumerate(range(0, n, batch_size)):
            b = order[s:s + batch_size]
            xb = X[b].astype(np.float32) / 255.0
            yb = y[b]

            loss, probs = model.train_step(xb, yb)
            total_loss += loss * len(b)
            correct += np.sum(probs.argmax(axis=1) == yb)

            if verbose and step % 200 == 0:
                done = s + len(b)
                print(f"\r  {done}/{n} samples | running loss {total_loss / done:.4f} "
                      f"| {time.time() - t0:.0f}s", end="", flush=True)

        val_loss, val_acc, val_top5 = evaluate(model, X, y, val_idx)
        model.set_lr(model.lr * LR_DECAY)

        if verbose:
            print(
                f"\rEpoch {epoch + 1}/{epochs} | Loss: {total_loss / n:.4f} | Acc: {correct / n:.2%} | "
                f"Val loss: {val_loss:.4f} | Val acc: {val_acc:.2%} | Val top-5: {val_top5:.2%} | "
                f"lr: {model.lr:.5f} | Time: {time.time() - t0:.1f}s" + " " * 10
            )

        if val_loss < model.best_val:
            model.best_val, bad = val_loss, 0
            model.save(model_path, categories)
            if verbose:
                print("  saved (best val loss)")
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print("Early stopping.")
                break


def main():
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "initialize"
    if mode not in ("initialize", "continue"):
        print("Usage: python TrainCNN.py [continue]")
        return

    categories = load_categories()
    if not categories:
        raise ValueError("No categories found in data/categories.txt")

    print("Loading cached data...")
    X, y = load_all(categories)
    print(f"X: {X.shape}, y: {y.shape}")

    # Same fixed-seed split as the MLP script -> directly comparable validation set
    rng = np.random.RandomState(0)
    perm = rng.permutation(len(X))
    n_val = int(len(X) * VAL_FRACTION)
    val_all = np.sort(perm[:n_val])
    val_idx = val_all[:VAL_SAMPLES]
    train_idx = perm[n_val:]

    if mode == "initialize":
        print("Initializing a new CNN...")
        model = CNN(len(categories))
    else:
        print("Resuming from saved CNN...")
        model, saved_categories = CNN.load(MODEL_PATH)
        if saved_categories != categories:
            raise ValueError("Saved model's categories don't match categories.txt")

    try:
        train(model, X, y, train_idx, val_idx, categories)
    except KeyboardInterrupt:
        print("\nInterrupted. Best checkpoint is saved; run 'python TrainCNN.py continue' to resume.")

    if MODEL_PATH.exists():
        best, _ = CNN.load(MODEL_PATH)
        v_loss, v_acc, v_top5 = evaluate(best, X, y, val_all)
        print(f"Best checkpoint on full validation split ({len(val_all)} samples): "
              f"loss {v_loss:.4f} | top-1 {v_acc:.2%} | top-5 {v_top5:.2%}")


if __name__ == "__main__":
    main()
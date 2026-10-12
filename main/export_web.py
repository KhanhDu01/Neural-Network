"""
export_web.py - turn the trained CNN checkpoint into static files the browser can use.

Outputs (into docs/ so GitHub Pages can serve them from the /docs folder):
    docs/model.json  - architecture, categories and tensor offsets/shapes
    docs/model.bin   - all weights/biases as raw little-endian float32, concatenated

The browser-side forward pass (docs/app.js) must match TrainCNN/CNN.py exactly:
    Conv3x3(1->C1) -> ReLU -> MaxPool2
    Conv3x3(C1->C2) -> ReLU -> MaxPool2
    Flatten -> Dense(HIDDEN) -> ReLU -> Dense(n_classes) -> Softmax

Usage:
    python main/export_web.py
"""
import json
from pathlib import Path

import numpy as np

MODEL_PATH = Path("main/model_cnn.npz")
OUT_DIR = Path("docs")

# Order matters: the browser reads these back-to-back from model.bin.
TENSORS = [
    "conv1_w", "conv1_b",
    "conv2_w", "conv2_b",
    "dense1_w", "dense1_b",
    "dense2_w", "dense2_b",
]


def main():
    z = np.load(MODEL_PATH, allow_pickle=False)

    c1, c2, hidden, n_classes = (int(v) for v in z["config"])
    categories = [str(c) for c in z["categories"]]

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    chunk = bytearray()
    meta = {}
    offset = 0  # running offset in float32 elements

    for name in TENSORS:
        arr = np.ascontiguousarray(z[name], dtype="<f4")
        meta[name] = {
            "offset": offset,
            "shape": list(arr.shape),
            "length": int(arr.size),
        }
        chunk += arr.tobytes()
        offset += int(arr.size)

    (OUT_DIR / "model.bin").write_bytes(bytes(chunk))

    model_json = {
        "conv1": {"in": 1, "out": c1, "k": 3, "pad": 1},
        "conv2": {"in": c1, "out": c2, "k": 3, "pad": 1},
        "hidden": hidden,
        "classes": n_classes,
        "input": [1, 28, 28],
        "totalFloats": offset,
        "categories": categories,
        "tensors": meta,
    }

    with open(OUT_DIR / "model.json", "w", encoding="utf-8") as f:
        json.dump(model_json, f, ensure_ascii=False, separators=(",", ":"))

    size_mb = len(chunk) / (1024 * 1024)
    print(f"Wrote {OUT_DIR / 'model.bin'} ({offset:,} floats, {size_mb:.2f} MB)")
    print(f"Wrote {OUT_DIR / 'model.json'} ({n_classes} categories, c1={c1}, c2={c2}, hidden={hidden})")


if __name__ == "__main__":
    main()

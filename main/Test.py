import json
from collections import Counter
from pathlib import Path

import numpy as np

from Layer import QuickDrawPreprocessor
from TrainCNN import CNN


# Paths (same layout as your old test script)
PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "main" / "model_cnn.npz"
TEST_DIR = PROJECT_DIR / "data" / "test"

BATCH_SIZE = 512

# The training cache holds the first N recognized drawings of every category
# (2,415,000 / 345 = 7,000 per class). If your test files are the SAME full files
# the model trained on, skip those drawings so the model is only scored on
# drawings it has never seen. If the test files are separate downloads of
# held-out data, set this to 0.
SKIP_TRAINED = 7000

# Cap per file (None = whole file). Useful if the files are huge and you just want a quick estimate.
MAX_PER_FILE = None


def category_from_filename(path):
    prefix = "full_simplified_"
    suffix = ".ndjson"

    name = path.name

    if not name.startswith(prefix) or not name.endswith(suffix):
        raise ValueError(f"Unexpected test filename: {name}")

    return name[len(prefix):-len(suffix)]


def to_uint8(img):
    """Match build_cache.py exactly: float [0, 1] -> uint8 -> (model divides by 255)."""
    return np.clip(img * 255, 0, 255).astype(np.uint8)


def evaluate_file(path, expected_index, model, categories, preprocessor,
                  skip=SKIP_TRAINED, max_per_file=MAX_PER_FILE):
    """Evaluate one category file without loading the entire file into RAM."""
    top1 = 0
    top5 = 0
    total = 0
    skipped = 0
    wrong = Counter()   # predicted class index -> count, for wrong top-1 predictions
    batch = []

    def flush():
        nonlocal top1, top5, total
        X = np.asarray(batch, dtype=np.uint8)
        probs = model.predict_proba(X)

        pred = probs.argmax(axis=1)
        k = min(5, probs.shape[1])
        best5 = np.argpartition(probs, -k, axis=1)[:, -k:]

        top1 += int(np.sum(pred == expected_index))
        top5 += int(np.sum((best5 == expected_index).any(axis=1)))
        total += len(pred)
        wrong.update(pred[pred != expected_index].tolist())
        batch.clear()

    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON in {path}, line {line_number}. "
                    "The file may be incomplete."
                ) from e

            if not record.get("recognized", True):
                continue

            if skipped < skip:
                skipped += 1
                continue

            batch.append(to_uint8(preprocessor.load_drawing(record)))

            if len(batch) >= BATCH_SIZE:
                flush()

            if max_per_file is not None and total + len(batch) >= max_per_file:
                break

    # Process the final, possibly smaller batch
    if batch:
        flush()

    top3_wrong = [(categories[i], n) for i, n in wrong.most_common(3)]
    return {
        "top1": top1,
        "top5": top5,
        "total": total,
        "acc": top1 / total if total else 0.0,
        "acc5": top5 / total if total else 0.0,
        "confused_with": top3_wrong,
    }


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\nRun TrainCNN.py first."
        )

    if not TEST_DIR.exists():
        raise FileNotFoundError(
            f"Test directory not found: {TEST_DIR}\n"
            "Create data/test/ and put held-out NDJSON files there."
        )

    model, categories = CNN.load(MODEL_PATH)
    preprocessor = QuickDrawPreprocessor()

    files = sorted(TEST_DIR.glob("full_simplified_*.ndjson"))

    if not files:
        print(f"No test files found in {TEST_DIR}")
        return

    total_top1 = 0
    total_top5 = 0
    total_samples = 0

    print(f"Loaded CNN with {len(categories)} output categories.")
    print(f"Evaluating {len(files)} test files "
          f"(skipping the first {SKIP_TRAINED:,} recognized drawings of each)...\n")

    for path in files:
        category = category_from_filename(path)

        if category not in categories:
            print(f"Skipping unknown category: {category}")
            continue

        expected_index = categories.index(category)

        r = evaluate_file(path, expected_index, model, categories, preprocessor)

        if r["total"] == 0:
            print(f"{category}: no drawings left after skipping {SKIP_TRAINED:,} "
                  f"- set SKIP_TRAINED lower if this file is held-out data\n")
            continue

        total_top1 += r["top1"]
        total_top5 += r["top5"]
        total_samples += r["total"]

        print(
            f"{category}: top-1 {r['acc']:.2%} | top-5 {r['acc5']:.2%} "
            f"({r['top1']}/{r['total']} correct)"
        )
        if r["confused_with"]:
            print("    most often mistaken for: " +
                  ", ".join(f"{c} ({n})" for c, n in r["confused_with"]))

    if total_samples:
        print("\n===== TEST RESULTS =====")
        print(f"Total drawings: {total_samples:,}")
        print(f"Top-1 correct: {total_top1:,} ({total_top1 / total_samples:.2%})")
        print(f"Top-5 correct: {total_top5:,} ({total_top5 / total_samples:.2%})")
    else:
        print("No eligible test drawings were found.")


if __name__ == "__main__":
    main()
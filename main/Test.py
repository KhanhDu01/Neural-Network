
import json
from pathlib import Path

import numpy as np
from Layer import QuickDrawPreprocessor


# Paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "main" / "model.json"
TEST_DIR = PROJECT_DIR / "data" / "test"

BATCH_SIZE = 512


def load_model():
    """Load saved weights and category mapping."""
    with open(MODEL_PATH, "r", encoding="utf-8") as f:
        params = json.load(f)

    if "categories" not in params:
        raise ValueError(
            "Model has no saved category mapping. "
            "Add model_params['categories'] = categories in save_model()."
        )

    weights = [
        np.asarray(params[f"weights_layer{i}"], dtype=np.float32)
        for i in range(1, 4)
    ]

    biases = [
        np.asarray(params[f"biases_layer{i}"], dtype=np.float32)
        for i in range(1, 4)
    ]

    categories = params["categories"]

    if weights[-1].shape[1] != len(categories):
        raise ValueError(
            "Output layer size does not match the saved category list."
        )

    return weights, biases, categories


def predict_batch(X, weights, biases):
    """Run forward propagation and return class probabilities."""
    X = X @ weights[0] + biases[0]
    X = np.maximum(X, 0)  # ReLU

    X = X @ weights[1] + biases[1]
    X = np.maximum(X, 0)  # ReLU

    logits = X @ weights[2] + biases[2]

    # Numerically stable softmax
    logits -= np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(logits)

    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def category_from_filename(path):
    prefix = "full_simplified_"
    suffix = ".ndjson"

    name = path.name

    if not name.startswith(prefix) or not name.endswith(suffix):
        raise ValueError(f"Unexpected test filename: {name}")

    return name[len(prefix):-len(suffix)]


def evaluate_file(path, expected_index, weights, biases, preprocessor):
    """Evaluate one category file without loading the entire file into RAM."""
    correct = 0
    total = 0
    batch = []

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

            batch.append(preprocessor.load_drawing(record))

            if len(batch) >= BATCH_SIZE:
                X = np.asarray(batch, dtype=np.float32)
                probabilities = predict_batch(X, weights, biases)

                predictions = np.argmax(probabilities, axis=1)
                correct += int(np.sum(predictions == expected_index))
                total += len(predictions)
                batch.clear()

    # Process the final, possibly smaller batch
    if batch:
        X = np.asarray(batch, dtype=np.float32)
        probabilities = predict_batch(X, weights, biases)

        predictions = np.argmax(probabilities, axis=1)
        correct += int(np.sum(predictions == expected_index))
        total += len(predictions)

    accuracy = correct / total if total else 0.0
    return correct, total, accuracy


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\nTrain and save your model first."
        )

    if not TEST_DIR.exists():
        raise FileNotFoundError(
            f"Test directory not found: {TEST_DIR}\n"
            "Create data/test/ and put held-out NDJSON files there."
        )

    weights, biases, categories = load_model()
    preprocessor = QuickDrawPreprocessor()

    files = sorted(TEST_DIR.glob("full_simplified_*.ndjson"))

    if not files:
        print(f"No test files found in {TEST_DIR}")
        return

    total_correct = 0
    total_samples = 0

    print(f"Loaded model with {len(categories)} output categories.")
    print(f"Evaluating {len(files)} test files...\n")

    for path in files:
        category = category_from_filename(path)

        if category not in categories:
            print(f"Skipping unknown category: {category}")
            continue

        expected_index = categories.index(category)

        correct, total, accuracy = evaluate_file(
            path,
            expected_index,
            weights,
            biases,
            preprocessor,
        )

        total_correct += correct
        total_samples += total

        print(
            f"{category}: {accuracy:.2%} "
            f"({correct}/{total} correct)"
        )

    if total_samples:
        print("\n===== TEST RESULTS =====")
        print(f"Total drawings: {total_samples:,}")
        print(f"Correct predictions: {total_correct:,}")
        print(f"Overall accuracy: {total_correct / total_samples:.2%}")
    else:
        print("No eligible test drawings were found.")


if __name__ == "__main__":
    main()
# build_cache.py
import json
import requests
import numpy as np
from pathlib import Path
from Layer import QuickDrawPreprocessor

DATA_DIR = Path("data")
CACHE_DIR = DATA_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
N_PER_CLASS = 2000
BASE_URL = "https://storage.googleapis.com/quickdraw_dataset/full/simplified"

def cache_category(category, pre):
    out = CACHE_DIR / f"{category}.npy"
    if out.exists():
        return

    url = f"{BASE_URL}/{requests.utils.quote(category)}.ndjson"
    samples = []
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            record = json.loads(line)
            if not record.get("recognized", True):
                continue
            img = pre.load_drawing(record)               # float in [0, 1]
            samples.append(np.clip(img * 255, 0, 255).astype(np.uint8))
            if len(samples) >= N_PER_CLASS:
                break

    np.save(out, np.stack(samples))
    print(f"Cached {category}: {len(samples)} samples")

if __name__ == "__main__":
    with open(DATA_DIR / "categories.txt", encoding="utf-8") as f:
        categories = [l.strip() for l in f if l.strip()]
    pre = QuickDrawPreprocessor()
    for c in categories:
        cache_category(c, pre)
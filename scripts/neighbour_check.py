"""Are consecutive image IDs more similar than random pairs? (crude thumbnail check)"""
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path("data/raw/rdd2022")
COUNTRIES = ["India", "Japan", "Czech", "United_States"]
rng = np.random.default_rng(0)


def thumb(path):
    return np.asarray(Image.open(path).convert("L").resize((32, 32)), dtype=np.float32)


for country in COUNTRIES:
    files = sorted((ROOT / country / "train" / "images").iterdir())

    mid = len(files) // 2
    window = files[mid:mid + 1500]
    t = np.stack([thumb(p) for p in window])
    consec = np.abs(t[1:] - t[:-1]).mean(axis=(1, 2))

    sample = rng.choice(len(files), size=1500, replace=False)
    r = np.stack([thumb(files[k]) for k in sample])
    i = rng.integers(0, len(r), 3000)
    j = rng.integers(0, len(r), 3000)
    keep = i != j
    rand = np.abs(r[i[keep]] - r[j[keep]]).mean(axis=(1, 2))

    cutoff = np.percentile(rand, 5)
    print(f"{country:>14}: median diff consecutive={np.median(consec):6.1f}  random={np.median(rand):6.1f}  "
          f"| consecutive pairs more similar than 95% of random pairs: {100 * (consec < cutoff).mean():5.1f}%")
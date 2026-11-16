import sys
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SIZE = 32
GROUP_COL = "group_97"
VAL_FRAC = 0.2
SEED = 0

def thumb(p):
    with Image.open(p) as im:
        im.draft("L", (SIZE * 2, SIZE * 2))
        im = im.convert("L").resize((SIZE, SIZE), Image.BILINEAR)
        return np.asarray(im, dtype=np.float32).ravel()

def load_vectors(country, ids):
    cache = ROOT / "data" / "processed" / f"thumbs_{country}.npy"
    if cache.exists():
        return np.load(cache)
    img_dir = ROOT / "data" / "raw" / "rdd2022" / country / "train" / "images"
    X = np.stack([thumb(img_dir / f"{i}.jpg") for i in ids])
    X -= X.mean(axis=1, keepdims=True)
    X /= X.std(axis=1, keepdims=True) + 1e-6
    X /= np.sqrt(X.shape[1])
    np.save(cache, X)
    return X

def max_corr(Xv, Xt):
    out = np.empty(len(Xv), dtype=np.float32)
    for s in range(0, len(Xv), 500):
        out[s:s + 500] = (Xv[s:s + 500] @ Xt.T).max(axis=1)
    return out

def report(name, mc, n_val):
    print(f"  {name} (val n={n_val}): median max-corr {np.median(mc):.3f}; "
          f"val images with a train image >=0.92: {(mc >= 0.92).mean():.1%}, "
          f">=0.95: {(mc >= 0.95).mean():.1%}, >=0.97: {(mc >= 0.97).mean():.1%}")

def main(country):
    df = pd.read_csv(ROOT / "data" / "processed" / f"groups_{country}.csv")
    X = load_vectors(country, df["image_id"])
    n = len(df)
    rng = np.random.default_rng(SEED)
    n_val = int(n * VAL_FRAC)
    print(f"\n{country}: {n} images")

    perm = rng.permutation(n)
    val, train = perm[:n_val], perm[n_val:]
    report("random split", max_corr(X[val], X[train]), len(val))

    sizes = df[GROUP_COL].value_counts()
    val_groups, count = set(), 0
    for g in rng.permutation(sizes.index.to_numpy()):
        if count >= n_val:
            break
        val_groups.add(g)
        count += sizes[g]
    is_val = df[GROUP_COL].isin(val_groups).to_numpy()
    report("group split (0.97)", max_corr(X[is_val], X[~is_val]), int(is_val.sum()))

if __name__ == "__main__":
    for c in sys.argv[1:]:
        main(c)
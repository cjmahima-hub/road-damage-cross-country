import sys
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SIZE = 32
THRESHOLDS = [0.97, 0.95, 0.92]      # strictest to loosest
MIN_THR = min(THRESHOLDS)

def thumb(p):
    with Image.open(p) as im:
        im.draft("L", (SIZE * 2, SIZE * 2))
        im = im.convert("L").resize((SIZE, SIZE), Image.BILINEAR)
        return np.asarray(im, dtype=np.float32).ravel()

def find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x

def main(country):
    img_dir = ROOT / "data" / "raw" / "rdd2022" / country / "train" / "images"
    paths = sorted(img_dir.glob("*.jpg"))
    n = len(paths)
    X = np.stack([thumb(p) for p in paths])
    X -= X.mean(axis=1, keepdims=True)
    X /= X.std(axis=1, keepdims=True) + 1e-6
    X /= np.sqrt(X.shape[1])

    # collect every pair (i < j) with correlation >= the loosest threshold
    I, J, V = [], [], []
    for s in range(0, n, 500):
        C = X[s:s + 500] @ X.T
        ii, jj = np.nonzero(C >= MIN_THR)
        ii = ii + s
        keep = ii < jj
        I.append(ii[keep]); J.append(jj[keep]); V.append(C[ii[keep] - s, jj[keep]])
    I, J, V = np.concatenate(I), np.concatenate(J), np.concatenate(V)
    print(f"\n{country}: {n} images, {len(I)} pairs with corr >= {MIN_THR}")

    out = pd.DataFrame({"country": country, "image_id": [p.stem for p in paths]})
    for thr in THRESHOLDS:
        parent = list(range(n))
        for a, b in zip(I[V >= thr], J[V >= thr]):
            ra, rb = find(parent, a), find(parent, b)
            if ra != rb:
                parent[ra] = rb
        roots = np.array([find(parent, i) for i in range(n)])
        sizes = pd.Series(roots).value_counts()
        multi = sizes[sizes >= 2]
        print(f"thr {thr}: groups of 2+: {len(multi)}, "
              f"images in them: {int(multi.sum())} ({multi.sum() / n:.1%}), "
              f"top 5 sizes: {sizes.head(5).tolist()}")
        out[f"group_{int(thr * 100)}"] = [f"{country}_g{r:05d}" for r in roots]

    out_csv = ROOT / "data" / "processed" / f"groups_{country}.csv"
    out.to_csv(out_csv, index=False)
    print("saved", out_csv)

if __name__ == "__main__":
    for c in sys.argv[1:]:
        main(c)
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SIZE = 32

def thumb(p):
    with Image.open(p) as im:
        im.draft("L", (SIZE * 2, SIZE * 2))
        im = im.convert("L").resize((SIZE, SIZE), Image.BILINEAR)
        return np.asarray(im, dtype=np.float32).ravel()

def main(country):
    img_dir = ROOT / "data" / "raw" / "rdd2022" / country / "train" / "images"
    paths = sorted(img_dir.glob("*.jpg"))
    X = np.stack([thumb(p) for p in paths])
    X -= X.mean(axis=1, keepdims=True)
    X /= X.std(axis=1, keepdims=True) + 1e-6
    X /= np.sqrt(X.shape[1])          # now X @ X.T = correlation matrix
    n = len(paths)
    best_i = np.zeros(n, dtype=int)
    best_c = np.zeros(n, dtype=np.float32)
    for s in range(0, n, 1000):
        C = X[s:s + 1000] @ X.T
        rows = np.arange(C.shape[0])
        C[rows, s + rows] = -1        # ignore self-match
        best_i[s:s + 1000] = C.argmax(axis=1)
        best_c[s:s + 1000] = C.max(axis=1)

    num = np.array([int(p.stem.split("_")[-1]) for p in paths])
    df = pd.DataFrame({
        "image": [p.stem for p in paths],
        "nearest": [paths[j].stem for j in best_i],
        "corr": best_c,
        "id_gap": np.abs(num - num[best_i]),
    })
    out_csv = ROOT / "data" / "processed" / f"nn_{country}.csv"
    df.to_csv(out_csv, index=False)

    print(country, "images:", n)
    print(df["corr"].describe(percentiles=[.5, .9, .99]).round(3))
    for t in (0.8, 0.9, 0.95):
        hi = df[df["corr"] > t]
        far = (hi["id_gap"] > 10).sum()
        print(f"corr > {t}: {len(hi)} images, of which nearest is >10 IDs away: {far}")

    # montage of the 8 most similar unique pairs
    seen, pairs = set(), []
    for k in df.sort_values("corr", ascending=False).index:
        key = tuple(sorted((k, int(best_i[k]))))
        if key not in seen:
            seen.add(key)
            pairs.append((k, int(best_i[k]), float(best_c[k])))
        if len(pairs) == 8:
            break
    fig, axes = plt.subplots(4, 4, figsize=(14, 14))
    for ax in axes.ravel():
        ax.axis("off")
    for n_, (a, b, c) in enumerate(pairs):
        r, col = divmod(n_, 2)
        for off, idx in enumerate((a, b)):
            ax = axes[r, col * 2 + off]
            ax.imshow(Image.open(paths[idx]))
            ax.set_title(f"{paths[idx].stem}\ncorr {c:.3f}", fontsize=8)
    out_png = ROOT / "reports" / "figures" / f"dupes_{country}.png"
    fig.tight_layout()
    fig.savefig(out_png, dpi=80)
    print("saved", out_csv, "and", out_png)

if __name__ == "__main__":
    main(sys.argv[1])
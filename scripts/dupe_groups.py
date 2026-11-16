import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
THRESHOLDS = [0.97, 0.95, 0.92, 0.90]

def find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x

def main(country):
    df = pd.read_csv(ROOT / "data" / "processed" / f"nn_{country}.csv")
    index = {name: i for i, name in enumerate(df["image"])}
    a = df["image"].map(index).to_numpy()
    b = df["nearest"].map(index).to_numpy()
    c = df["corr"].to_numpy()
    n = len(df)
    print(f"\n{country}: {n} images")
    for thr in THRESHOLDS:
        parent = list(range(n))
        for i in np.flatnonzero(c >= thr):
            ra, rb = find(parent, a[i]), find(parent, b[i])
            if ra != rb:
                parent[ra] = rb
        roots = np.array([find(parent, i) for i in range(n)])
        sizes = pd.Series(roots).value_counts()
        multi = sizes[sizes >= 2]
        print(
            f"thr {thr}: groups of 2+: {len(multi)}, "
            f"images in them: {int(multi.sum())} ({multi.sum() / n:.1%}), "
            f"largest group: {int(sizes.max())}, "
            f"sizes 2/3/4-5/6-10/11+: "
            f"{(multi == 2).sum()}/{(multi == 3).sum()}/"
            f"{((multi >= 4) & (multi <= 5)).sum()}/"
            f"{((multi >= 6) & (multi <= 10)).sum()}/{(multi >= 11).sum()}"
        )

if __name__ == "__main__":
    for country in sys.argv[1:]:
        main(country)
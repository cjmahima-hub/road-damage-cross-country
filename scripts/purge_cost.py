import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GROUP_COL = "group_97"
VAL_FRAC = 0.2
SEED = 0

def main(country):
    df = pd.read_csv(ROOT / "data" / "processed" / f"groups_{country}.csv")
    X = np.load(ROOT / "data" / "processed" / f"thumbs_{country}.npy")
    n = len(df)
    rng = np.random.default_rng(SEED)
    n_val = int(n * VAL_FRAC)

    sizes = df[GROUP_COL].value_counts()
    val_groups, count = set(), 0
    for g in rng.permutation(sizes.index.to_numpy()):
        if count >= n_val:
            break
        val_groups.add(g)
        count += sizes[g]
    is_val = df[GROUP_COL].isin(val_groups).to_numpy()
    Xv, Xt = X[is_val], X[~is_val]

    # for each train image: its highest correlation with any val image
    mc = np.empty(len(Xt), dtype=np.float32)
    for s in range(0, len(Xt), 500):
        mc[s:s + 500] = (Xt[s:s + 500] @ Xv.T).max(axis=1)

    print(f"\n{country}: val {len(Xv)}, train before purge {len(Xt)}")
    for thr in (0.97, 0.95, 0.92):
        keep = mc < thr
        print(f"  purge train images with corr >= {thr} to any val image: "
              f"removes {int((~keep).sum())}, train left {int(keep.sum())} ({keep.mean():.1%})")

if __name__ == "__main__":
    for c in sys.argv[1:]:
        main(c)
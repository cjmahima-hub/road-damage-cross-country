import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
GROUP_COL = "group_97"
TEST_FRAC, VAL_FRAC = 0.20, 0.10
PURGE_THR = 0.95
SEED = 0
CLASSES = ["D00", "D10", "D20", "D40"]
ORDER = ["train", "val", "test", "purged"]

def max_corr(A, B):
    out = np.empty(len(A), dtype=np.float32)
    for s in range(0, len(A), 500):
        out[s:s + 500] = (A[s:s + 500] @ B.T).max(axis=1)
    return out

def main(country, boxes):
    df = pd.read_csv(PROC / f"groups_{country}.csv")
    X = np.load(PROC / f"thumbs_{country}.npy")
    n = len(df)
    rng = np.random.default_rng(SEED)

    sizes = df[GROUP_COL].value_counts()
    assign, n_test, n_val = {}, 0, 0
    for g in rng.permutation(sizes.index.to_numpy()):
        if n_test < TEST_FRAC * n:
            assign[g] = "test"; n_test += sizes[g]
        elif n_val < VAL_FRAC * n:
            assign[g] = "val"; n_val += sizes[g]
        else:
            assign[g] = "train"
    df["split"] = df[GROUP_COL].map(assign)

    is_test = (df["split"] == "test").to_numpy()
    is_val = (df["split"] == "val").to_numpy()
    is_train = (df["split"] == "train").to_numpy()
    purge = np.zeros(n, dtype=bool)
    tr = np.flatnonzero(is_train)
    purge[tr] = max_corr(X[tr], X[is_val | is_test]) >= PURGE_THR
    va = np.flatnonzero(is_val)
    purge[va] = max_corr(X[va], X[is_test]) >= PURGE_THR
    df.loc[purge, "split"] = "purged"

    b = boxes[boxes["country"] == country]
    b = b.merge(df[["image_id", "split"]], on="image_id", how="left")
    print(f"\n{country}: {n} images, {len(b)} boxes after dropping degenerate ones, "
          f"boxes with no matching image: {int(b['split'].isna().sum())}")
    print("images per split:")
    print(df["split"].value_counts().reindex(ORDER).fillna(0).astype(int).to_string())
    tab = (b.pivot_table(index="split", columns="class", values="image_id",
                         aggfunc="count", fill_value=0)
             .reindex(ORDER).reindex(columns=CLASSES).fillna(0).astype(int))
    print("boxes per class per split:")
    print(tab.to_string())

    out = PROC / f"splits_{country}.csv"
    df[["country", "image_id", "split"]].to_csv(out, index=False)
    print("saved", out)

if __name__ == "__main__":
    boxes = pd.read_csv(PROC / "boxes_4class.csv")
    boxes = boxes[(boxes["xmax"] > boxes["xmin"]) & (boxes["ymax"] > boxes["ymin"])]
    for c in sys.argv[1:]:
        main(c, boxes)
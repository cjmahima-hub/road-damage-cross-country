import sys
from pathlib import Path
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
BANDS = [(0.99, 1.01), (0.97, 0.99), (0.95, 0.97), (0.92, 0.95), (0.88, 0.92)]
PAIRS = 4

def main(country):
    df = pd.read_csv(ROOT / "data" / "processed" / f"nn_{country}.csv")
    img_dir = ROOT / "data" / "raw" / "rdd2022" / country / "train" / "images"
    fig, axes = plt.subplots(len(BANDS), PAIRS * 2, figsize=(20, 11))
    for ax in axes.ravel():
        ax.axis("off")
    for r, (lo, hi) in enumerate(BANDS):
        band = df[(df["corr"] >= lo) & (df["corr"] < hi)]
        print(f"corr {lo}-{hi}: {len(band)} images")
        if len(band) == 0:
            continue
        pick = band.sample(min(PAIRS, len(band)), random_state=0)
        for c, (_, row) in enumerate(pick.iterrows()):
            for off, name in enumerate((row["image"], row["nearest"])):
                ax = axes[r, c * 2 + off]
                ax.imshow(Image.open(img_dir / f"{name}.jpg"))
                ax.set_title(f"{name}\n{row['corr']:.3f}", fontsize=7)
    fig.tight_layout()
    out = ROOT / "reports" / "figures" / f"bands_{country}.png"
    fig.savefig(out, dpi=70)
    print("saved", out)

if __name__ == "__main__":
    main(sys.argv[1])
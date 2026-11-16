from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
ORDER = ["Czech", "India", "Japan", "United_States"]
df = pd.read_csv(ROOT / "results" / "results.csv")
# equal-size matrix; Czech's full run is already 1,588 images
eq = df[(df.setting == "equal1588") | ((df.train == "Czech") & (df.setting == "full"))]
M = eq.pivot(index="train", columns="test", values="map50").reindex(index=ORDER, columns=ORDER)
print(M.round(3).to_string())

fig, ax = plt.subplots(figsize=(6.5, 5))
im = ax.imshow(M.values, cmap="viridis", vmin=0)
ax.set_xticks(range(4), ORDER, rotation=30, ha="right")
ax.set_yticks(range(4), ORDER)
ax.set_xlabel("Tested on")
ax.set_ylabel("Trained on (1,588 images)")
ax.set_title("YOLO11n mAP50, test splits (seed 0)")
for i in range(4):
    for j in range(4):
        v = M.values[i, j]
        ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                color="white" if v < 0.25 else "black")
fig.colorbar(im, ax=ax)
fig.tight_layout()
out = ROOT / "docs" / "matrix_equal1588.png"
fig.savefig(out, dpi=150)
print("saved", out)
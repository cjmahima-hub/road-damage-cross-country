"""Draw boxes on sample images to eyeball label quality and neighbour similarity."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

ROOT = Path("data/raw/rdd2022")
OUT = Path("reports/figures")
OUT.mkdir(parents=True, exist_ok=True)
COLORS = {"D00": "yellow", "D10": "cyan", "D20": "magenta", "D40": "red"}
COUNTRIES = ["India", "Japan", "Czech", "United_States"]

df = pd.read_csv("data/processed/boxes_4class.csv")
df["w"] = df.xmax - df.xmin
df["h"] = df.ymax - df.ymin
df["rel_area"] = df.w * df.h / (df.img_w * df.img_h)

bad = df[(df.w <= 0) | (df.h <= 0)]
print("Zero-size boxes:")
print(bad[["country", "image_id", "class", "xmin", "ymin", "xmax", "ymax"]])


def find_image(country, image_id):
    return next((ROOT / country / "train" / "images").glob(f"{image_id}.*"))


def draw(ax, country, image_id):
    ax.imshow(Image.open(find_image(country, image_id)))
    ax.axis("off")
    for _, r in df[(df.country == country) & (df.image_id == image_id)].iterrows():
        ax.add_patch(patches.Rectangle(
            (r.xmin, r.ymin), r.w, r.h, fill=False,
            edgecolor=COLORS[r["class"]], linewidth=2))
    ax.set_title(image_id, fontsize=8)


def grid(country, image_ids, filename):
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for ax, image_id in zip(axes.flat, image_ids):
        draw(ax, country, image_id)
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=90)
    plt.close(fig)


for country in COUNTRIES:
    sub = df[df.country == country]
    all_ids = sorted(p.stem for p in (ROOT / country / "train" / "images").iterdir())

    random_ids = sub["image_id"].drop_duplicates().sample(6, random_state=0).tolist()
    largest = sub.sort_values("rel_area", ascending=False)["image_id"].drop_duplicates().head(6).tolist()
    start = len(all_ids) // 2
    consecutive = all_ids[start:start + 6]

    grid(country, random_ids, f"{country}_random.png")
    grid(country, largest, f"{country}_largest.png")
    grid(country, consecutive, f"{country}_consecutive.png")
    print(country, "-> consecutive ids:", consecutive)

print("Figures saved in", OUT)
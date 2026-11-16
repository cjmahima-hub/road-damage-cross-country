"""Keep the 4 target classes, validate boxes, and print per-country statistics."""
import pandas as pd

KEEP = ["D00", "D10", "D20", "D40"]
TOTAL_XMLS = {"India": 7706, "Japan": 10506, "Czech": 2829, "United_States": 4805}

df = pd.read_csv("data/processed/boxes.csv")
coords = ["xmin", "ymin", "xmax", "ymax", "img_w", "img_h"]
for c in coords:
    df[c] = pd.to_numeric(df[c], errors="coerce")
print("rows:", len(df), "| rows with missing numbers:", df[coords].isna().any(axis=1).sum())

extra = df[~df["class"].isin(KEEP)]
print("\nDropped non-target boxes:")
print(extra.groupby(["country", "class"]).size())
df = df[df["class"].isin(KEEP)].copy()

df["w"] = df.xmax - df.xmin
df["h"] = df.ymax - df.ymin
bad_size = (df.w <= 0) | (df.h <= 0)
oob = (df.xmin < 0) | (df.ymin < 0) | (df.xmax > df.img_w) | (df.ymax > df.img_h)
print("\nnon-positive-size boxes:", int(bad_size.sum()), "| out-of-bounds boxes:", int(oob.sum()))
if (bad_size | oob).any():
    print(df[bad_size | oob].groupby("country").size())

df["rel_area"] = (df.w * df.h) / (df.img_w * df.img_h)
print("\nBox size as a fraction of image area:")
print(df.groupby("country")["rel_area"].describe()[["count", "min", "25%", "50%", "75%", "max"]].round(4))

table = df.pivot_table(index="country", columns="class", values="image_id", aggfunc="count", fill_value=0)
print("\nBoxes per class:")
print(table)
print("\nClass share (%):")
print((table.div(table.sum(axis=1), axis=0) * 100).round(1))

with_box = df.groupby("country")["image_id"].nunique()
print("\nImages with at least one target box:")
for country, total in TOTAL_XMLS.items():
    n = int(with_box.get(country, 0))
    print(f"  {country:>14}: {n} of {total} ({100 * n / total:.1f}%)")

df.to_csv("data/processed/boxes_4class.csv", index=False)
print("\nWrote data/processed/boxes_4class.csv")
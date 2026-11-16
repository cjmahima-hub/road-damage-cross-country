import sys
from pathlib import Path
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "yolo"
CLS = {"D00": 0, "D10": 1, "D20": 2, "D40": 3}
MAXSIDE = 640

def main(country, boxes):
    sp = pd.read_csv(ROOT / "splits" / f"splits_{country}.csv")
    sp = sp[sp["split"].isin(["train", "val", "test"])]
    by_img = {k: g for k, g in boxes[boxes["country"] == country].groupby("image_id")}
    src = ROOT / "data" / "raw" / "rdd2022" / country / "train" / "images"
    n_img, n_box = {}, {}
    for _, r in sp.iterrows():
        s, iid = r["split"], r["image_id"]
        (OUT / country / "images" / s).mkdir(parents=True, exist_ok=True)
        (OUT / country / "labels" / s).mkdir(parents=True, exist_ok=True)
        with Image.open(src / f"{iid}.jpg") as im:
            im = im.convert("RGB")
            scale = MAXSIDE / max(im.size)
            if scale < 1:
                im = im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS)
            im.save(OUT / country / "images" / s / f"{iid}.jpg", quality=95)
        lines = []
        g = by_img.get(iid)
        if g is not None:
            for _, x in g.iterrows():
                x0, x1 = max(0, x["xmin"]), min(x["img_w"], x["xmax"])
                y0, y1 = max(0, x["ymin"]), min(x["img_h"], x["ymax"])
                if x1 <= x0 or y1 <= y0:
                    continue
                xc, yc = (x0 + x1) / 2 / x["img_w"], (y0 + y1) / 2 / x["img_h"]
                w, h = (x1 - x0) / x["img_w"], (y1 - y0) / x["img_h"]
                lines.append(f"{CLS[x['class']]} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
        (OUT / country / "labels" / s / f"{iid}.txt").write_text("\n".join(lines))
        n_img[s] = n_img.get(s, 0) + 1
        n_box[s] = n_box.get(s, 0) + len(lines)
    print(country, "images:", n_img, "boxes:", n_box)

if __name__ == "__main__":
    boxes = pd.read_csv(ROOT / "data" / "processed" / "boxes_4class.csv")
    boxes = boxes[(boxes["xmax"] > boxes["xmin"]) & (boxes["ymax"] > boxes["ymin"])]
    for c in sys.argv[1:]:
        main(c, boxes)
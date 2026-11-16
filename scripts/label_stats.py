"""Parse Pascal VOC XMLs for the four countries: write one row per box and print stats."""
import csv
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

ROOT = Path("data/raw/rdd2022")
OUT_CSV = Path("data/processed/boxes.csv")
COUNTRIES = ["India", "Japan", "Czech", "United_States"]

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

with open(OUT_CSV, "w", newline="") as fh:
    writer = csv.writer(fh)
    writer.writerow(["country", "image_id", "class", "xmin", "ymin", "xmax", "ymax", "img_w", "img_h"])

    for country in COUNTRIES:
        xml_dir = ROOT / country / "train" / "annotations" / "xmls"
        classes = Counter()
        sizes = Counter()
        n_images = n_empty = n_bad = 0

        for xml_file in sorted(xml_dir.glob("*.xml")):
            n_images += 1
            try:
                root = ET.parse(xml_file).getroot()
            except ET.ParseError:
                n_bad += 1
                continue

            w = root.findtext("size/width")
            h = root.findtext("size/height")
            sizes[(w, h)] += 1
            objects = root.findall("object")
            if not objects:
                n_empty += 1

            for obj in objects:
                name = (obj.findtext("name") or "").strip()
                classes[name] += 1
                box = obj.find("bndbox")
                coords = [box.findtext(k) for k in ("xmin", "ymin", "xmax", "ymax")]
                writer.writerow([country, xml_file.stem, name, *coords, w, h])

        print(f"\n=== {country} ===")
        print(f"train XMLs: {n_images}   unreadable: {n_bad}   images with no boxes: {n_empty}")
        print(f"total boxes: {sum(classes.values())}")
        for name, n in classes.most_common():
            print(f"   {name!r:>8}: {n}")
        print("most common image sizes:", sizes.most_common(3))

print(f"\nBox table written to {OUT_CSV}")
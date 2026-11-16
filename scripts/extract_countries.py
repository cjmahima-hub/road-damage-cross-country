"""Extract only India, Japan, Czech and United_States from the big RDD2022 zip."""
import shutil
import zipfile
from collections import Counter
from pathlib import Path

OUTER = Path("data/raw/RDD2022_released_through_CRDDC2022.zip")
INNER_DIR = Path("data/raw/_inner_zips")
OUT_DIR = Path("data/raw/rdd2022")
WANTED = ["India", "Japan", "Czech", "United_States"]

INNER_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile(OUTER) as outer:
    for country in WANTED:
        member = f"RDD2022/{country}.zip"
        inner_path = INNER_DIR / f"{country}.zip"

        if not inner_path.exists():
            print(f"[{country}] copying inner zip out of the big zip...")
            with outer.open(member) as src, open(inner_path, "wb") as dst:
                shutil.copyfileobj(src, dst)

        with zipfile.ZipFile(inner_path) as inner:
            bad = inner.testzip()
            if bad is not None:
                print(f"[{country}] CORRUPT file inside zip: {bad}")
                continue

            counts = Counter()
            for info in inner.infolist():
                if not info.is_dir():
                    counts["/".join(info.filename.split("/")[:-1])] += 1

            print(f"[{country}] integrity check passed. Files per folder:")
            for folder, n in sorted(counts.items()):
                print(f"   {n:>6}  {folder}")

            inner.extractall(OUT_DIR)
            print(f"[{country}] extracted into {OUT_DIR}")
"""List what is inside the RDD2022 zip without extracting anything."""
import zipfile
from collections import Counter
from pathlib import Path

ZIP_PATH = Path("data/raw/RDD2022_released_through_CRDDC2022.zip")

counts = Counter()
examples = {}

with zipfile.ZipFile(ZIP_PATH) as zf:
    infos = zf.infolist()
    print("Total entries:", len(infos))
    for info in infos:
        if info.is_dir():
            continue
        parts = info.filename.split("/")
        folder = tuple(parts[:-1])
        counts[folder] += 1
        examples.setdefault(folder, parts[-1])

for folder, n in sorted(counts.items()):
    print(f"{n:>7}  {'/'.join(folder)}   e.g. {examples[folder]}")
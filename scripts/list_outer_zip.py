"""Show the entries (inner zips) inside the big RDD2022 zip, with sizes."""
import zipfile
from pathlib import Path

ZIP_PATH = Path("data/raw/RDD2022_released_through_CRDDC2022.zip")

with zipfile.ZipFile(ZIP_PATH) as zf:
    for info in zf.infolist():
        print(f"{info.file_size / 1024**2:>9.1f} MB  {info.filename}")
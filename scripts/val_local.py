import sys
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
weights = ROOT / "models" / (sys.argv[1] if len(sys.argv) > 1 else "japan_best.pt")
country = sys.argv[2] if len(sys.argv) > 2 else "Japan"

yml = ROOT / "data" / "yolo" / f"{country}_eval.yaml"
yml.write_text(
    f"path: {(ROOT / 'data' / 'yolo' / country).as_posix()}\n"
    "train: images/train\nval: images/val\ntest: images/test\n"
    "names:\n  0: D00\n  1: D10\n  2: D20\n  3: D40\n"
)
m = YOLO(str(weights)).val(data=str(yml), split="test", imgsz=640, device="cpu",
                           batch=16, workers=0, plots=False, verbose=False)
print(f"{country} test: mAP50 {m.box.map50:.4f}  mAP50-95 {m.box.map:.4f}")
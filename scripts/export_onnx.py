from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
out = YOLO(str(ROOT / "models" / "japan_best.pt")).export(
    format="onnx", imgsz=640, opset=17, simplify=True, dynamic=False, half=False)
print("exported", out)
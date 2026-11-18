import sys, time, platform
from pathlib import Path
import numpy as np
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
imgs = sorted((ROOT / "data" / "yolo" / "Japan" / "images" / "test").glob("*.jpg"))[:60]
model = YOLO(str(ROOT / "models" / (sys.argv[1] if len(sys.argv) > 1 else "japan_best.pt")),
             task="detect")
print("python", platform.python_version(), "torch", torch.__version__,
      "threads", torch.get_num_threads())

for p in imgs[:5]:                      # warm-up, not timed
    model.predict(str(p), imgsz=640, device="cpu", verbose=False)

pre, inf, post, total = [], [], [], []
for p in imgs[5:]:
    t0 = time.perf_counter()
    r = model.predict(str(p), imgsz=640, device="cpu", verbose=False)[0]
    total.append((time.perf_counter() - t0) * 1000)
    pre.append(r.speed["preprocess"])
    inf.append(r.speed["inference"])
    post.append(r.speed["postprocess"])

med = lambda x: round(float(np.median(x)), 1)
print(f"images timed: {len(total)}")
print(f"median ms  preprocess {med(pre)}  inference {med(inf)}  "
      f"postprocess {med(post)}  end-to-end {med(total)}")
out = ROOT / "reports" / "figures" / "demo_japan.jpg"
r.save(filename=str(out))
print("boxes on last image:", len(r.boxes), "| saved", out)

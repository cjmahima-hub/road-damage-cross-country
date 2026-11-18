import numpy as np, torch, cv2, onnxruntime as ort
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
files = sorted((ROOT / "data" / "yolo" / "Japan" / "images" / "test").glob("*.jpg"))[:20]

def prep(p, size=640):
    img = cv2.imread(str(p))
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = round(h * r), round(w * r)
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    out = np.full((size, size, 3), 114, np.uint8)
    t, l = (size - nh) // 2, (size - nw) // 2
    out[t:t + nh, l:l + nw] = img
    return np.ascontiguousarray(out[:, :, ::-1].transpose(2, 0, 1)[None]).astype(np.float32) / 255.0

net = YOLO(str(ROOT / "models" / "japan_best.pt")).model.float().fuse().eval()
sess = ort.InferenceSession(str(ROOT / "models" / "japan_best.onnx"),
                            providers=["CPUExecutionProvider"])
name = sess.get_inputs()[0].name
worst_box, worst_cls = 0.0, 0.0
for p in files:
    x = prep(p)
    with torch.no_grad():
        a = net(torch.from_numpy(x))
    a = (a[0] if isinstance(a, (list, tuple)) else a).numpy()
    b = sess.run(None, {name: x})[0]
    d = np.abs(a - b)
    worst_box = max(worst_box, float(d[:, :4].max()))
    worst_cls = max(worst_cls, float(d[:, 4:].max()))
print("output shapes", a.shape, b.shape)
print(f"max abs diff over {len(files)} images: boxes (pixels) {worst_box:.5f}  class scores {worst_cls:.6f}")

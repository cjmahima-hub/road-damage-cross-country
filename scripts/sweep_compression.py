import csv, random, shutil, time
from pathlib import Path
import numpy as np, cv2, onnx, onnxruntime as ort
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO
from onnxruntime.quantization import (quantize_static, CalibrationDataReader,
                                      QuantType, QuantFormat)

ROOT = Path(__file__).resolve().parents[1]
SIZES = [320, 416, 512, 640]
N_CAL = 100
WORK = ROOT / "models" / "sweep"
YAML = ROOT / "data" / "yolo" / "Japan_eval.yaml"
TEST = sorted((ROOT / "data" / "yolo" / "Japan" / "images" / "test").glob("*.jpg"))[:60]
TRAIN = sorted((ROOT / "data" / "yolo" / "Japan" / "images" / "train").glob("*.jpg"))

def prep(p, size):
    img = cv2.imread(str(p))
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = round(h * r), round(w * r)
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    out = np.full((size, size, 3), 114, dtype=np.uint8)
    t, l = (size - nh) // 2, (size - nw) // 2
    out[t:t + nh, l:l + nw] = img
    return np.ascontiguousarray(out[:, :, ::-1].transpose(2, 0, 1)[None]).astype(np.float32) / 255.0

class Reader(CalibrationDataReader):
    def __init__(self, files, name, size):
        self.it, self.name, self.size = iter(files), name, size
    def get_next(self):
        p = next(self.it, None)
        return None if p is None else {self.name: prep(p, self.size)}

def quantize(src, dst, size):
    model = onnx.load(str(src))
    exclude = [n.name for n in model.graph.node if n.name.startswith("/model.23/")]
    assert exclude, "detect head not found"
    files = random.Random(0).sample(TRAIN, N_CAL)   # train split only
    name = ort.InferenceSession(str(src), providers=["CPUExecutionProvider"]).get_inputs()[0].name
    quantize_static(str(src), str(dst), Reader(files, name, size),
                    quant_format=QuantFormat.QDQ, per_channel=True,
                    weight_type=QuantType.QInt8, activation_type=QuantType.QUInt8,
                    nodes_to_exclude=exclude)
    q = onnx.load(str(dst))
    del q.metadata_props[:]
    q.metadata_props.extend(model.metadata_props)
    onnx.save(q, str(dst))

def latency(path, size):
    m = YOLO(str(path), task="detect")
    for p in TEST[:5]:
        m.predict(str(p), imgsz=size, device="cpu", verbose=False)
    inf, tot = [], []
    for p in TEST[5:]:
        t0 = time.perf_counter()
        r = m.predict(str(p), imgsz=size, device="cpu", verbose=False)[0]
        tot.append((time.perf_counter() - t0) * 1000)
        inf.append(r.speed["inference"])
    return float(np.median(inf)), float(np.median(tot))

def accuracy(path, size):
    m = YOLO(str(path), task="detect").val(data=str(YAML), split="test", imgsz=size,
                                          device="cpu", batch=1, workers=0,
                                          plots=False, verbose=False)
    return float(m.box.map50), float(m.box.map)

rows = [["model", "imgsz", "size_mb", "map50", "map50_95", "infer_ms", "e2e_ms"]]
WORK.mkdir(parents=True, exist_ok=True)
for s in SIZES:
    d = WORK / f"s{s}"
    d.mkdir(exist_ok=True)
    shutil.copy(ROOT / "models" / "japan_best.pt", d / "japan_best.pt")
    fp32 = Path(YOLO(str(d / "japan_best.pt")).export(
        format="onnx", imgsz=s, opset=17, simplify=True, dynamic=False))
    int8 = d / "japan_best_int8_headfp32.onnx"
    quantize(fp32, int8, s)
    for label, path in [("onnx_fp32", fp32), ("onnx_int8_headfp32", int8)]:
        inf, tot = latency(path, s)
        m50, m5095 = accuracy(path, s)
        rows.append([label, s, round(path.stat().st_size / 1e6, 1),
                     round(m50, 4), round(m5095, 4), round(inf, 1), round(tot, 1)])
        print("DONE", rows[-1], flush=True)

with open(ROOT / "results" / "compression.csv", "w", newline="") as f:
    csv.writer(f).writerows(rows)

fig, ax = plt.subplots(figsize=(7, 5))
for label, mk in [("onnx_fp32", "o"), ("onnx_int8_headfp32", "s")]:
    pts = [r for r in rows[1:] if r[0] == label]
    ax.plot([r[6] for r in pts], [r[3] for r in pts], marker=mk, label=label)
    for r in pts:
        ax.annotate(str(r[1]), (r[6], r[3]), textcoords="offset points",
                    xytext=(4, 4), fontsize=8)
ax.set_xlabel("end-to-end ms per image (CPU, batch 1)")
ax.set_ylabel("mAP50 (Japan test)")
ax.set_title("Japan-trained YOLO11n: accuracy vs CPU latency\n(labels = input size)")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(ROOT / "docs" / "accuracy_vs_latency.png", dpi=150)
print("saved results\\compression.csv and docs\\accuracy_vs_latency.png")

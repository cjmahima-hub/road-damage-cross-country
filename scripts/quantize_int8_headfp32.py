import random
from pathlib import Path
import numpy as np
import cv2
import onnx
import onnxruntime as ort
from onnxruntime.quantization import (quantize_static, CalibrationDataReader,
                                      QuantType, QuantFormat)

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "models" / "japan_best.onnx"
DST = ROOT / "models" / "japan_best_int8_headfp32.onnx"
N_CAL = 100

def prep(p, size=640):
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
    def __init__(self, files, input_name):
        self.it, self.name = iter(files), input_name
    def get_next(self):
        p = next(self.it, None)
        return None if p is None else {self.name: prep(p)}

model = onnx.load(str(SRC))
exclude = [n.name for n in model.graph.node if n.name.startswith("/model.23/")]
print("nodes excluded (detect head):", len(exclude), "of", len(model.graph.node))
if not exclude:
    print("last 15 nodes:")
    for n in model.graph.node[-15:]:
        print(" ", n.op_type, n.name)
    raise SystemExit("could not find the detect head by name; paste the lines above")

# calibration images come from the TRAIN split only (never val/test)
train = sorted((ROOT / "data" / "yolo" / "Japan" / "images" / "train").glob("*.jpg"))
files = random.Random(0).sample(train, N_CAL)
name = ort.InferenceSession(str(SRC), providers=["CPUExecutionProvider"]).get_inputs()[0].name

quantize_static(str(SRC), str(DST), Reader(files, name),
                quant_format=QuantFormat.QDQ, per_channel=True,
                weight_type=QuantType.QInt8, activation_type=QuantType.QUInt8,
                nodes_to_exclude=exclude)

q = onnx.load(str(DST))
del q.metadata_props[:]
q.metadata_props.extend(model.metadata_props)
onnx.save(q, str(DST))
print(f"FP32 {SRC.stat().st_size / 1e6:.1f} MB -> INT8 (head FP32) {DST.stat().st_size / 1e6:.1f} MB")

def best_score(path, imgs):
    s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    n = s.get_inputs()[0].name
    return max(float(s.run(None, {n: prep(p)})[0][:, 4:].max()) for p in imgs)

five = files[:5]
print(f"highest class score on 5 train images: FP32 {best_score(SRC, five):.3f}"
      f"  INT8-head-FP32 {best_score(DST, five):.3f}")

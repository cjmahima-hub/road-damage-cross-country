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
DST = ROOT / "models" / "japan_best_int8.onnx"
N_CAL = 100

def letterbox(img, size=640):
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = round(h * r), round(w * r)
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    out = np.full((size, size, 3), 114, dtype=np.uint8)
    t, l = (size - nh) // 2, (size - nw) // 2
    out[t:t + nh, l:l + nw] = img
    return out

class Reader(CalibrationDataReader):
    def __init__(self, files, input_name):
        self.it, self.name = iter(files), input_name
    def get_next(self):
        p = next(self.it, None)
        if p is None:
            return None
        x = letterbox(cv2.imread(str(p)))[:, :, ::-1].transpose(2, 0, 1)[None]
        return {self.name: np.ascontiguousarray(x).astype(np.float32) / 255.0}

# calibration images come from the TRAIN split only (never val/test)
train = sorted((ROOT / "data" / "yolo" / "Japan" / "images" / "train").glob("*.jpg"))
files = random.Random(0).sample(train, N_CAL)
name = ort.InferenceSession(str(SRC), providers=["CPUExecutionProvider"]).get_inputs()[0].name

quantize_static(str(SRC), str(DST), Reader(files, name),
                quant_format=QuantFormat.QDQ, per_channel=True,
                weight_type=QuantType.QInt8, activation_type=QuantType.QUInt8)

# keep the Ultralytics metadata (class names, stride, image size) on the INT8 file
src, q = onnx.load(str(SRC)), onnx.load(str(DST))
del q.metadata_props[:]
q.metadata_props.extend(src.metadata_props)
onnx.save(q, str(DST))
print(f"FP32 {SRC.stat().st_size / 1e6:.1f} MB -> INT8 {DST.stat().st_size / 1e6:.1f} MB")

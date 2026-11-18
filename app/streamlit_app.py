import io, tempfile, time
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
CLASSES = {0: "D00 longitudinal crack", 1: "D10 transverse crack",
           2: "D20 alligator crack", 3: "D40 pothole"}
MAX_FRAMES = 40

st.set_page_config(page_title="Road damage detector", layout="wide")
st.title("Road damage detection: FP32 vs INT8 on CPU")
st.caption("YOLO11n trained on 1,588 Japan (RDD2022) images. It scores far lower on other countries "
           "(see the cross-country matrix in the repo), so expect misses on roads outside Japan.")

size = st.sidebar.selectbox("Input size", [320, 416, 512, 640], index=3)
conf = st.sidebar.slider("Confidence threshold", 0.05, 0.9, 0.25, 0.05)
every = st.sidebar.slider("Video: analyse every Nth frame", 1, 30, 10)

@st.cache_resource
def load(kind, size):
    name = "japan_best.onnx" if kind == "fp32" else "japan_best_int8_headfp32.onnx"
    return YOLO(str(ROOT / "models" / "sweep" / f"s{size}" / name), task="detect")

def get_frames(data, is_image, every):
    if is_image:
        return [np.array(Image.open(io.BytesIO(data)).convert("RGB"))[:, :, ::-1].copy()]
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(data)
    cap = cv2.VideoCapture(tmp.name)
    frames, i = [], 0
    while len(frames) < MAX_FRAMES:
        ok, f = cap.read()
        if not ok:
            break
        if i % every == 0:
            frames.append(f)
        i += 1
    cap.release()
    Path(tmp.name).unlink(missing_ok=True)
    return frames

def run(model, frame, size, conf):
    t0 = time.perf_counter()
    r = model.predict(frame, imgsz=size, conf=conf, device="cpu", verbose=False)[0]
    ms = (time.perf_counter() - t0) * 1000
    counts = Counter(int(c) for c in r.boxes.cls.tolist())
    return cv2.cvtColor(r.plot(), cv2.COLOR_BGR2RGB), ms, counts

@st.cache_data(show_spinner="Running both models...")
def analyse(data, is_image, size, conf, every):
    frames = get_frames(data, is_image, every)
    if not frames:
        return None
    fp32, int8 = load("fp32", size), load("int8", size)
    for m in (fp32, int8):                      # warm-up so the first timing is fair
        m.predict(frames[0], imgsz=size, conf=conf, device="cpu", verbose=False)
    out = {"fp32": [], "int8": []}
    for f in frames:
        out["fp32"].append(run(fp32, f, size, conf))
        out["int8"].append(run(int8, f, size, conf))
    return out

file = st.file_uploader("Upload a road image or a short dashcam clip",
                        type=["jpg", "jpeg", "png", "mp4", "mov"])
if file is None:
    st.info("Upload a file to start.")
    st.stop()

res = analyse(file.getvalue(), file.type.startswith("image"), size, conf, every)
if res is None:
    st.error("Could not read any frames from this file.")
    st.stop()

n = len(res["fp32"])
idx = (st.slider("Frame", 1, n, 1) - 1) if n > 1 else 0
c1, c2 = st.columns(2)
for col, key, title in [(c1, "fp32", "FP32 (ONNX)"), (c2, "int8", "INT8 (ONNX, Detect head FP32)")]:
    img, ms, counts = res[key][idx]
    col.subheader(title)
    col.image(img)
    col.write(f"{ms:.1f} ms this frame | "
              + (", ".join(f"{CLASSES[k]}: {v}" for k, v in sorted(counts.items())) or "no detections"))

st.divider()
rows = []
for key, name in [("fp32", "FP32"), ("int8", "INT8")]:
    tot = Counter()
    for r in res[key]:
        tot.update(r[2])
    rows.append({"model": name,
                 "median ms/frame": round(float(np.median([r[1] for r in res[key]])), 1),
                 **{CLASSES[k].split()[0]: tot.get(k, 0) for k in CLASSES}})
st.write(f"Summary over {n} analysed frame(s) at {size}px, confidence {conf}")
st.table(rows)
st.caption("Timings include pre/post-processing from a single pass on this machine; "
           "for repeated measurements see results/latency_repeats.csv.")

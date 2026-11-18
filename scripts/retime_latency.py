import csv, time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
SIZES = [320, 416, 512, 640]
LABELS = {"onnx_fp32": "japan_best.onnx", "onnx_int8_headfp32": "japan_best_int8_headfp32.onnx"}
REPS = 5
TEST = sorted((ROOT / "data" / "yolo" / "Japan" / "images" / "test").glob("*.jpg"))[:60]

models = {(l, s): YOLO(str(ROOT / "models" / "sweep" / f"s{s}" / f), task="detect")
          for s in SIZES for l, f in LABELS.items()}
for (l, s), m in models.items():                      # warm-up, not timed
    for p in TEST[:5]:
        m.predict(str(p), imgsz=s, device="cpu", verbose=False)

e2e = {k: [] for k in models}
inf = {k: [] for k in models}
for rep in range(REPS):                               # interleaved so drift hits all models alike
    for s in SIZES:
        for l in LABELS:
            m = models[(l, s)]
            t_all, t_inf = [], []
            for p in TEST[5:]:
                t0 = time.perf_counter()
                r = m.predict(str(p), imgsz=s, device="cpu", verbose=False)[0]
                t_all.append((time.perf_counter() - t0) * 1000)
                t_inf.append(r.speed["inference"])
            e2e[(l, s)].append(float(np.median(t_all)))
            inf[(l, s)].append(float(np.median(t_inf)))
    print("repeat", rep + 1, "of", REPS, "done", flush=True)

acc = {}
with open(ROOT / "results" / "compression.csv") as f:
    for row in csv.DictReader(f):
        acc[(row["model"], int(row["imgsz"]))] = float(row["map50"])

rows = [["model", "imgsz", "map50", "e2e_median_ms", "e2e_min_ms", "e2e_max_ms", "infer_median_ms"]]
for s in SIZES:
    for l in LABELS:
        k = (l, s)
        rows.append([l, s, acc[k], round(float(np.median(e2e[k])), 1), round(min(e2e[k]), 1),
                     round(max(e2e[k]), 1), round(float(np.median(inf[k])), 1)])
with open(ROOT / "results" / "latency_repeats.csv", "w", newline="") as f:
    csv.writer(f).writerows(rows)
for r in rows:
    print(*r, sep="\t")

fig, ax = plt.subplots(figsize=(7, 5))
for l, mk in [("onnx_fp32", "o"), ("onnx_int8_headfp32", "s")]:
    pts = [r for r in rows[1:] if r[0] == l]
    ax.errorbar([r[3] for r in pts], [r[2] for r in pts],
                xerr=[[r[3] - r[4] for r in pts], [r[5] - r[3] for r in pts]],
                marker=mk, capsize=3, label=l)
    for r in pts:
        ax.annotate(str(r[1]), (r[3], r[2]), textcoords="offset points", xytext=(4, 4), fontsize=8)
ax.set_xlabel("end-to-end ms per image (median of 5 repeats; bars = min-max)")
ax.set_ylabel("mAP50 (Japan test)")
ax.set_title("Japan-trained YOLO11n: accuracy vs CPU latency\n(labels = input size)")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(ROOT / "docs" / "accuracy_vs_latency.png", dpi=150)
print("saved results\\latency_repeats.csv and docs\\accuracy_vs_latency.png")

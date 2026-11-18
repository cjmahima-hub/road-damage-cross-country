# What didn't work

- Plain static INT8 (QDQ, per-channel, 100 train-split calibration images) of japan_best.onnx gave
  mAP50 0.0000 on the Japan test split (ONNX FP32: 0.3750). The model was smaller (3.2 MB) and faster, but useless.
- Excluding the 84 Detect-head nodes (of 320) from quantisation restored accuracy: mAP50 0.3655, mAP50-95 0.1554.
- ONNX FP32 scores lower than PyTorch FP32 (0.3750 vs 0.3986 mAP50) although raw outputs agree to
  0.0025 px on 20 images. Cause not established (suspect: different letterboxing in validation).
- Full-size (all train images) cross-country runs were not done: Colab sessions were too slow.

- Size sweep (320/416/512/640): single-pass latency timings showed ONNX INT8 *slower* than FP32 at 320 and 416
  (25.8 vs 15.3 ms and 28.2 vs 24.0 ms end-to-end). Re-timing with 5 interleaved repeats showed INT8 faster at
  every size (medians 13.9 vs 15.2, 19.3 vs 21.2, 25.9 vs 28.9, 34.6 vs 38.8 ms). The single pass was affected by
  a machine slowdown (cause not established), so those timings were discarded. Accuracy numbers were unaffected.

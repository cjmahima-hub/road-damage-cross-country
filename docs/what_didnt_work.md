# What didn't work

- Plain static INT8 (QDQ, per-channel, 100 train-split calibration images) of japan_best.onnx gave
  mAP50 0.0000 on the Japan test split (ONNX FP32: 0.3750). The model was smaller (3.2 MB) and faster, but useless.
- Excluding the 84 Detect-head nodes (of 320) from quantisation restored accuracy: mAP50 0.3655, mAP50-95 0.1554.
- ONNX FP32 scores lower than PyTorch FP32 (0.3750 vs 0.3986 mAP50) although raw outputs agree to
  0.0025 px on 20 images. Cause not established (suspect: different letterboxing in validation).
- Full-size (all train images) cross-country runs were not done: Colab sessions were too slow.

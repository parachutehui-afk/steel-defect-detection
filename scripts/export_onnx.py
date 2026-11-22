#!/usr/bin/env python3
"""Export a trained YOLO model to ONNX and benchmark CPU inference.

Usage:
    venv/bin/python scripts/export_onnx.py [--weights runs/train/.../weights/best.pt] [--imgsz 640]
"""
import argparse
import platform
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "runs" / "train" / "yolov8n_neu-det" / "weights" / "best.pt"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--n", type=int, default=100, help="images for the benchmark")
    args = ap.parse_args()

    model = YOLO(args.weights)
    onnx_path = model.export(format="onnx", imgsz=args.imgsz)
    print(f"ONNX exported: {onnx_path}")

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    x = np.random.rand(1, 3, args.imgsz, args.imgsz).astype(np.float32)
    name = sess.get_inputs()[0].name
    for _ in range(5):  # warmup
        sess.run(None, {name: x})
    t0 = time.perf_counter()
    for _ in range(args.n):
        sess.run(None, {name: x})
    ms = (time.perf_counter() - t0) / args.n * 1000
    print(f"CPU: {platform.processor() or 'unknown'}")
    print(f"ONNX inference: {ms:.1f} ms/image (n={args.n}, imgsz={args.imgsz}, ONNX Runtime, 1 thread-batch)")


if __name__ == "__main__":
    main()

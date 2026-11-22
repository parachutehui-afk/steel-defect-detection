#!/usr/bin/env python3
"""Evaluate a trained YOLO model on the NEU-DET test split.

Prints mAP50, mAP50-95 and per-class precision/recall, and saves the
confusion matrix plot next to the weights.

Usage:
    venv/bin/python scripts/evaluate.py [--weights runs/train/yolov8n_neu-det/weights/best.pt]
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "runs" / "train" / "yolov8n_neu-det" / "weights" / "best.pt"))
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    model = YOLO(args.weights)
    metrics = model.val(
        data=str(ROOT / "data" / "neu_det.yaml"),
        split="test",
        imgsz=args.imgsz,
        device="cpu",
        plots=True,
        save_json=True,
    )
    print(f"mAP50    : {metrics.box.map50:.4f}")
    print(f"mAP50-95 : {metrics.box.map:.4f}")
    print(f"mean P   : {metrics.box.mp:.4f}  mean R: {metrics.box.mr:.4f}")
    print("per-class (P / R / mAP50 / mAP50-95):")
    for i, name in enumerate(metrics.names.values()):
        print(
            f"  {name:15s} P={metrics.box.p[i]:.3f}  R={metrics.box.r[i]:.3f}  "
            f"mAP50={metrics.box.ap50[i]:.3f}  mAP50-95={metrics.box.ap[i]:.3f}"
        )
    print(f"plots saved under: {Path(metrics.save_dir)}")


if __name__ == "__main__":
    main()

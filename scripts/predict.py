#!/usr/bin/env python3
"""Run inference on one image (or a folder) and save the annotated result.

Usage:
    venv/bin/python scripts/predict.py image.jpg [--weights ...] [--out assets/]
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="image file or folder")
    ap.add_argument("--weights", default=str(ROOT / "runs" / "train" / "yolov8n_neu-det" / "weights" / "best.pt"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out", default=str(ROOT / "assets" / "predictions"))
    args = ap.parse_args()

    model = YOLO(args.weights)
    results = model.predict(
        source=args.input, imgsz=args.imgsz, conf=args.conf,
        device="cpu", save=True, project=args.out, name="", exist_ok=True,
    )
    for r in results:
        boxes = [(r.names[int(b.cls)], float(b.conf)) for b in r.boxes] if r.boxes is not None else []
        print(f"{r.path}: {boxes}")
    print(f"annotated images saved to {args.out}")


if __name__ == "__main__":
    main()

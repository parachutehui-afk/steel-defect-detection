#!/usr/bin/env python3
"""Train YOLOv8n on NEU-DET (CPU).

Usage:
    venv/bin/python scripts/train.py [--imgsz 640] [--epochs 50] [--batch 16] [--model yolov8n.pt]
"""
import argparse
import time
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--project", default=str(ROOT / "runs" / "train"))
    ap.add_argument("--name", default="yolov8n_neu-det")
    args = ap.parse_args()

    t0 = time.time()
    model = YOLO(args.model)
    model.train(
        data=str(ROOT / "data" / "neu_det.yaml"),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device="cpu",
        project=args.project,
        name=args.name,
        exist_ok=True,
        seed=42,
    )
    print(f"training took {(time.time() - t0) / 60:.1f} min")
    print(f"best weights: {args.project}/{args.name}/weights/best.pt")


if __name__ == "__main__":
    main()

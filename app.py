"""Streamlit demo: upload a steel-surface image, get defect boxes.

Runs the trained YOLOv8n ONNX model (CPU, ONNX Runtime).
"""
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent
ONNX_PATH = ROOT / "runs" / "train" / "yolov8n_neu-det" / "weights" / "best.onnx"
CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
IMGSZ = 416  # must match the imgsz used for training and ONNX export

COLORS = [(255, 0, 0), (0, 180, 0), (0, 120, 255), (255, 165, 0), (160, 32, 240), (0, 200, 200)]


@st.cache_resource
def load_model():
    return ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])


def letterbox(img, size=IMGSZ):
    h, w = img.shape[:2]
    s = min(size / w, size / h)
    nw, nh = int(w * s), int(h * s)
    resized = cv2.resize(img, (nw, nh))
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[(size - nh) // 2:(size - nh) // 2 + nh, (size - nw) // 2:(size - nw) // 2 + nw] = resized
    return canvas, s, (size - nw) // 2, (size - nh) // 2


def predict(sess, img_bgr):
    canvas, s, dx, dy = letterbox(img_bgr)
    x = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    t0 = time.perf_counter()
    out = sess.run(None, {sess.get_inputs()[0].name: x})[0][0]
    ms = (time.perf_counter() - t0) * 1000
    # out: (84, N) -> boxes xywh + 6 class scores (YOLOv8 ONNX)
    boxes = out[:4].T
    scores = out[4:].T
    cls = scores.argmax(axis=1)
    conf = scores.max(axis=1)
    keep = conf > 0.25
    boxes, cls, conf = boxes[keep], cls[keep], conf[keep]
    # xywh (640-space) -> xyxy (original image space)
    x1 = (boxes[:, 0] - boxes[:, 2] / 2 - dx) / s
    y1 = (boxes[:, 1] - boxes[:, 3] / 2 - dy) / s
    x2 = (boxes[:, 0] + boxes[:, 2] / 2 - dx) / s
    y2 = (boxes[:, 1] + boxes[:, 3] / 2 - dy) / s
    # simple NMS via cv2
    idx = cv2.dnn.NMSBoxes(
        np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist(),
        conf.tolist(), 0.25, 0.45,
    )
    dets = []
    for i in (idx.flatten() if len(idx) else []):
        dets.append((CLASSES[int(cls[i])], float(conf[i]), int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i])))
    return dets, ms


st.set_page_config(page_title="NEU-DET 钢材表面缺陷检测", layout="centered")
st.title("钢材表面缺陷检测 · NEU-DET + YOLOv8n")
st.caption("上传一张钢板表面图，模型会框出缺陷并给出类别与置信度。CPU 推理（ONNX）。")

if not ONNX_PATH.exists():
    st.error("未找到 ONNX 模型，请先运行 scripts/train.py 与 scripts/export_onnx.py。")
    st.stop()

sess = load_model()
up = st.file_uploader("选择图片（jpg / png）", type=["jpg", "jpeg", "png"])
if up is not None:
    img = np.array(Image.open(up).convert("RGB"))
    img_bgr = img[:, :, ::-1].copy()
    dets, ms = predict(sess, img_bgr)
    for name, c, x1, y1, x2, y2 in dets:
        col = COLORS[CLASSES.index(name)]
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), col, 2)
        cv2.putText(img_bgr, f"{name} {c:.2f}", (x1, max(y1 - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    st.image(img_bgr[:, :, ::-1], caption=f"检出 {len(dets)} 处缺陷 · 推理 {ms:.0f} ms", use_container_width=True)
    if dets:
        st.table([{"类别": n, "置信度": f"{c:.2f}", "位置 (x1,y1,x2,y2)": f"({x1},{y1},{x2},{y2})"}
                  for n, c, x1, y1, x2, y2 in dets])
    else:
        st.info("未检出缺陷（置信度阈值 0.25）。")
    st.caption(f"模型：YOLOv8n @ NEU-DET · 推理耗时 {ms:.0f} ms/张（ONNX Runtime, CPU）")

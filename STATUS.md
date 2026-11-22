# Demo 1 build status — DONE 2026-11-21 16:35 UTC

## Final results (test split, 180 images, conf 0.25)
- mAP50 **0.748**, mAP50-95 **0.440**, mean P 0.633, mean R 0.763
- Per class mAP50: crazing 0.419, inclusion 0.865, patches 0.881,
  pitted_surface 0.786, rolled-in_scale 0.588, scratches 0.945
- ONNX inference: **120.9 ms/image** (n=100, imgsz 416, ONNX Runtime, AMD EPYC 7D13)
- Streamlit app verified end-to-end via AppTest (upload -> detections, no errors)
  and HTTP 200 from the live server; fixed one real bug (app used IMGSZ 640,
  ONNX was exported at 416).
- Local git repo initialized, initial commit 50307f0, clean tree. NO remote
  added — user must create GitHub repo `steel-defect-detection` himself,
  then push from here.
- README.md covers all 6 required items, with 3 real bad cases in assets/.

## Build log (for reference)

- 2026-11-21: environment ready — venv Python 3.12, torch 2.6.0+cpu (Aliyun
  mirror; download.pytorch.org blocked), torchvision 0.21.0+cpu, ultralytics
  8.4.216, onnx/onnxruntime, streamlit, opencv-headless.
- Dataset: HF KeenForgeAI/NEU-DET-corrected, 1,797 images / 1,797 XML / 1,797
  YOLO labels, 6 classes. Split (dataset's own ImageSets): train 1437 /
  val 180 / test 180 (8:1:1). Prepared to data/yolo + data/neu_det.yaml.
- Training: YOLOv8n, imgsz 416 (NOT 640 — one epoch at 640 took >6 min and
  was still not done, 50 epochs would be ~8h; at 416 one epoch is ~8 min,
  50 epochs ~6.5h), epochs 50, batch 16, workers 2, CPU.
  Background session proc_96f4f653b925, log runs/train416.log.
  Started 2026-11-21 ~09:59 UTC; expected finish ~16:30 UTC.
  Note: first launch crashed after epoch 1 (val mAP50 0.389) because
  ultralytics 8.4.216 needs `polars` to save checkpoints — polars 2.1.0
  installed, relaunched from scratch. Epoch 1 confirmed at 50% 10:03 UTC.
- 2026-11-21 ~11:19 UTC: VM RESTARTED (uptime 0 min) — training killed by
  SIGTERM mid-epoch-11 (86%). Progress saved: epochs 1-10 done
  (epoch 10: val mAP50 0.571, mAP50-95 0.282), best.pt + last.pt intact.
  Resumed 11:21 UTC from last.pt (epoch 9) via detached setsid+nohup
  (pid 1713, log runs/resume.log), watcher session proc_7843c32f124f
  (scripts/watch_train.sh) will report when it finishes.
  40 epochs left at ~7.5 min/epoch ≈ 5h, expected finish ~16:30 UTC.
  If the VM restarts again, resume the same way from last.pt.
- TODO after training: evaluate.py (test split), export_onnx.py + CPU
  ms/image benchmark, predict.py sanity check, app.py smoke test,
  README.md (6 required items), bad-case images in assets/.
- Do NOT create a GitHub repo; user creates steel-defect-detection himself.

#!/bin/bash
# Watch the detached training process (pid passed as $1) and report when it ends.
# Exits when the process is gone; prints the final metrics rows.
PID="$1"
PROJ="$HOME/workspace/goals/industrial-computer-vision-freelancing/demo-1-steel-defect"
while kill -0 "$PID" 2>/dev/null; do
  sleep 300
done
echo "TRAINING PROCESS $PID EXITED at $(date -u)"
tail -3 "$PROJ/runs/train/yolov8n_neu-det/results.csv"
ls -la "$PROJ/runs/train/yolov8n_neu-det/weights/"
tr '\r' '\n' < "$PROJ/runs/resume.log" | tail -5

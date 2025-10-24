#!/usr/bin/env python3
import os, time, csv, signal
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO
import torch

# ---------- config (env variables overrides) ----------
INTERVAL_SEC = int(os.getenv("SV_INTERVAL_SEC", "30"))
SRC          = os.getenv("SV_CAM_SRC", "0")   # "0" for /dev/video0, or rtsp/http URL
WEIGHTS      = os.getenv("SV_WEIGHTS", "yolov8m.pt")  # yolov8n.pt on Le Potato
IMG_SIZE     = int(os.getenv("SV_IMG_SIZE", "640"))
CONF         = float(os.getenv("SV_CONF", "0.35"))
IOU          = float(os.getenv("SV_IOU", "0.5"))
CLASSES      = [0, 24, 56, 60]  # person, backpack, chair, dining table

# Paths (from job_bashrc)
HOME      = Path(os.getenv("HOME", ".")).resolve()
ROOT      = HOME
YOLO_DIR  = ROOT / "YOLO"
OUT_DIR   = YOLO_DIR / "runs" / "cam_preds"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH  = OUT_DIR / "detections.csv"
SAVE_JPGS = os.getenv("SV_SAVE_JPGS", "1") == "1"   # save annotated frames

DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

# ---------- exit ----------
_running = True
def _stop(*_):
    global _running
    _running = False
signal.signal(signal.SIGINT, _stop)
signal.signal(signal.SIGTERM, _stop)

# ---------- helpers ----------
def ts_name() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def ensure_csv_header(p: Path):
    if not p.exists():
        with p.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timestamp","image","class","conf","x1","y1","x2","y2"])

# ---------- main ----------
def main():
    # Open camera
    src = int(SRC) if SRC.isdigit() else SRC
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera source: {SRC}")

    # Load model
    model = YOLO(str(YOLO_DIR / WEIGHTS))
    ensure_csv_header(CSV_PATH)

    print(f"[SeatView] Starting cam loop (interval={INTERVAL_SEC}s, device={DEVICE})")
    print(f"[SeatView] Writing to {OUT_DIR}")

    while _running:
        t0 = time.time()
        ok, frame = cap.read()
        if not ok or frame is None:
            print("[SeatView] WARN: failed to read frame; retrying next interval")
        else:
            stamp = ts_name()
            # Run prediction in-memory
            results = model.predict(
                frame,
                imgsz=IMG_SIZE,
                conf=CONF,
                iou=IOU,
                classes=CLASSES,
                device=DEVICE,
                verbose=False
            )
            r = results[0]
            names = r.names

            # raw boxes + save annotated image for audit
            img_name = f"{stamp}.jpg"
            if SAVE_JPGS:
                draw = frame.copy()
                for b in r.boxes:
                    cls = int(b.cls[0]); conf = float(b.conf[0])
                    x1,y1,x2,y2 = map(int, b.xyxy[0])
                    cv2.rectangle(draw, (x1,y1), (x2,y2), (0,255,0), 2)
                    cv2.putText(draw, f"{names[cls]} {conf:.2f}", (x1, y1-6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                cv2.imwrite(str(OUT_DIR / img_name), draw)

            # Append detections to CSV
            with CSV_PATH.open("a", newline="") as f:
                w = csv.writer(f)
                for b in r.boxes:
                    cls = int(b.cls[0]); conf = float(b.conf[0])
                    x1,y1,x2,y2 = [float(v) for v in b.xyxy[0].tolist()]
                    w.writerow([stamp, img_name if SAVE_JPGS else "", names[cls], f"{conf:.4f}",
                                f"{x1:.1f}", f"{y1:.1f}", f"{x2:.1f}", f"{y2:.1f}"])

            # You can trigger your existing chair_table_prox.py on the CSV or read it directly.
            print(f"[SeatView] {stamp}: {len(r.boxes)} detections")

        # sleep the remainder to maintain fixed cadence
        elapsed = time.time() - t0
        to_sleep = max(0.0, INTERVAL_SEC - elapsed)
        time.sleep(to_sleep)

    cap.release()
    print("[SeatView] Stopped.")

if __name__ == "__main__":
    main()

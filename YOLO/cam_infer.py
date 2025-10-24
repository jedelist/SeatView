#!/usr/bin/env python3
# cam_infer.py — SeatView: webcam → YOLO → CSV (+optional annotated JPG)
# Works on Le Potato (CPU) with Logitech C920 @ /dev/video1

import os, time, csv, signal, subprocess, shlex
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# ----------------------- CONFIG (via env) -----------------------
INTERVAL_SEC = int(os.getenv("SV_INTERVAL_SEC", "30"))
SRC_ENV      = os.getenv("SV_CAM_SRC", "/dev/video1")  # "1" or "/dev/video1"
WEIGHTS      = os.getenv("SV_WEIGHTS", "yolov8n.pt")   # start small on CPU
IMG_SIZE     = int(os.getenv("SV_IMG_SIZE", "512"))
CONF         = float(os.getenv("SV_CONF", "0.35"))
IOU          = float(os.getenv("SV_IOU", "0.5"))
SAVE_JPGS    = os.getenv("SV_SAVE_JPGS", "1") == "1"

# Classes: person(0), backpack(24), chair(56), dining table(60)
CLASSES      = [0, 24, 56, 60]

# Project paths (repo root assumed to be $HOME/SeatView)
HOME      = Path(os.getenv("HOME", ".")).resolve()
ROOT      = HOME / "seatview" / "SeatView"
YOLO_DIR  = ROOT / "YOLO"
OUT_DIR   = YOLO_DIR / "runs" / "cam_preds"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH  = OUT_DIR / "detections.csv"

# Device selection
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

# Camera capture settings (hint MJPEG @ 1280x720, low FPS to save CPU)
CAP_WIDTH  = int(os.getenv("SV_CAP_WIDTH",  "1280"))
CAP_HEIGHT = int(os.getenv("SV_CAP_HEIGHT", "720"))
CAP_FPS    = int(os.getenv("SV_CAP_FPS",    "5"))

# ----------------------- SIGNAL HANDLING ------------------------
_running = True
def _stop(*_):
    global _running
    _running = False
signal.signal(signal.SIGINT, _stop)
signal.signal(signal.SIGTERM, _stop)

# ----------------------- HELPERS --------------------------------
def ts_name() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def ensure_csv_header(p: Path):
    if not p.exists():
        with p.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timestamp","image","class","conf","x1","y1","x2","y2"])

def open_camera_from_env(src_env: str):
    """
    OpenCV open with correct backend:
      - numeric index -> CAP_V4L2
      - device path   -> CAP_ANY (avoid 'capture by name' warning)
    Then hint MJPEG 1280x720 at CAP_FPS.
    """
    if src_env.isdigit():
        cap = cv2.VideoCapture(int(src_env), cv2.CAP_V4L2)
    else:
        cap = cv2.VideoCapture(src_env, cv2.CAP_ANY)

    # Hint MJPEG + size + fps (C920 handles MJPG in-hardware, saves CPU)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAP_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          CAP_FPS)

    # Warmup reads (some cams need a couple frames)
    if cap.isOpened():
        for _ in range(3):
            ok, _frame = cap.read()
            if ok:
                break
    return cap if cap.isOpened() else None

def grab_frame_ffmpeg(dev_path: str):
    """
    Robust fallback: pull one MJPEG frame with ffmpeg and decode to ndarray.
    Avoids disk I/O; everything stays in-memory.
    """
    cmd = (
        f"ffmpeg -hide_banner -loglevel error "
        f"-f video4linux2 -input_format mjpeg -video_size {CAP_WIDTH}x{CAP_HEIGHT} "
        f"-i {dev_path} -frames:v 1 -f mjpeg -"
    )
    p = subprocess.run(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=8)
    if p.returncode != 0 or not p.stdout:
        return None
    arr = np.frombuffer(p.stdout, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

def device_path_from_src(src_env: str) -> str:
    return src_env if not src_env.isdigit() else f"/dev/video{src_env}"

# ----------------------- MAIN -----------------------------------
def main():
    # Ultralytics sometimes wants a writable config dir on SBCs
    os.environ.setdefault("YOLO_CONFIG_DIR", str(HOME / ".config" / "Ultralytics"))
    (HOME / ".config" / "Ultralytics").mkdir(parents=True, exist_ok=True)

    # Load model
    model_path = str((YOLO_DIR / WEIGHTS) if not Path(WEIGHTS).exists() else WEIGHTS)
    model = YOLO(model_path)

    # Outputs
    ensure_csv_header(CSV_PATH)

    # Open came

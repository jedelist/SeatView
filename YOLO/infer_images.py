from ultralytics import YOLO
import torch, csv
from pathlib import Path

# ---------- config ----------
SOURCE_DIR   = "images_jpg"        # put your iPhone photos here (jpg/png); convert HEIC first
OUT_ROOT     = "runs/seatv_preds"
RUN_NAME     = "yolo8m_filtered"
WEIGHTS      = "yolov8m.pt"        # try yolov8l.pt for more accuracy
IMG_SIZE     = 960                 # bump to 1280 if chairs are tiny
CONF         = 0.35
IOU          = 0.5
# COCO ids we care about
CLASSES      = [0, 24, 56, 60]     # person, backpack, chair, dining table
# -----------------------------

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

model = YOLO(WEIGHTS)

results = model.predict(
    source=SOURCE_DIR,
    imgsz=IMG_SIZE,
    conf=CONF,
    iou=IOU,
    classes=CLASSES,
    device=DEVICE,
    save=True,                 # saves annotated images
    project=OUT_ROOT,
    name=RUN_NAME,
    exist_ok=True
)

# write a CSV of detections
out_dir  = Path(OUT_ROOT) / RUN_NAME
csv_path = out_dir / "detections.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["image","class","conf","x1","y1","x2","y2"])
    for r in results:
        names = r.names
        imname = Path(r.path).name
        for b in r.boxes:
            cls = int(b.cls[0]); conf = float(b.conf[0])
            x1,y1,x2,y2 = [float(v) for v in b.xyxy[0].tolist()]
            w.writerow([imname, names[cls], f"{conf:.4f}", f"{x1:.1f}", f"{y1:.1f}", f"{x2:.1f}", f"{y2:.1f}"])

print(f"✅ Done. Annotated images: {out_dir}\n📄 CSV: {csv_path}")

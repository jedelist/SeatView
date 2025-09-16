from ultralytics import YOLO
import cv2

# COCO dataset pretrained wights
model = YOLO("yolov8s.pt")  # or 'n' for nano

# Classes to keep: person(0), backpack(24), chair(56), dining table(60)
# (Optionally include handbag(26), suitcase(28) if useful)
CLASSES = [0, 24, 56, 60]  # potentially extend

src = 0  # webcam; replace with "rtsp://..." or a video path
cap = cv2.VideoCapture(src)

while True:
    ok, frame = cap.read()
    if not ok: break

    results = model.predict(frame, imgsz=640, conf=0.35, iou=0.5, classes=CLASSES, device=0)
    r = results[0]

    # Draw results in boxes
    for b in r.boxes:
        cls = int(b.cls[0]); conf = float(b.conf[0])
        x1,y1,x2,y2 = map(int, b.xyxy[0])
        label = f"{r.names[cls]} {conf:.2f}"
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(frame, label, (x1, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    cv2.imshow("filtered", frame)
    if cv2.waitKey(1) == 27: break  # ESC to quit

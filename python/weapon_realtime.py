# ✅ Real-Time Weapon Detection with YOLOv8 + Webcam

import cv2
import numpy as np
from ultralytics import YOLO
from norfair import Detection, Tracker
import time
import csv
import random
from pathlib import Path

# Resolve project paths
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent if BASE_DIR.name == 'python' else BASE_DIR

# ✅ Load YOLOv8 weapon detection model
model = YOLO(str((ROOT_DIR / "models" / "best.pt").resolve()))
model.fuse()
model.conf = 0.3  # increase confidence for more accuracy

# ✅ Convert YOLO detections to Norfair format
def yolo_to_norfair(results):
    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        detections.append(Detection(points=np.array([cx, cy]), scores=np.array([box.conf.item()])))
    return detections

# ✅ Initialize Norfair tracker
tracker = Tracker(distance_function="euclidean", distance_threshold=30)

# ✅ Color helper
colors = {}
def get_color(label):
    if label not in colors:
        colors[label] = tuple(random.randint(50, 255) for _ in range(3))
    return colors[label]

# ✅ Prepare CSV log file
output_dir = (ROOT_DIR / "output").resolve()
output_dir.mkdir(parents=True, exist_ok=True)
log_file = str(output_dir / "realtime_weapon_log.csv")
with open(log_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Timestamp", "Weapon", "TrackID"])

detected_ids = set()

# ✅ Start webcam (0 = default)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Webcam not found or can't be opened.")
    exit()

print("🚨 Press 'q' to quit...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(source=frame, verbose=False)[0]
    detections = yolo_to_norfair(results)
    tracked_objects = tracker.update(detections=detections)

    for trk in tracked_objects:
        cx, cy = trk.estimate[0]
        track_id = trk.id
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            box_cx = (x1 + x2) / 2
            if abs(box_cx - cx) < 20:
                cls = int(box.cls[0])
                label = results.names[cls]
                color = get_color(label)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} ID:{track_id}", (x1, max(10, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Logging & Alert
                if track_id not in detected_ids:
                    print(f"[ALERT] {label} detected - ID {track_id}")
                    detected_ids.add(track_id)
                    with open(log_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), label, track_id])
                break

    cv2.imshow("🔴 Real-Time Weapon Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Detection ended. Log saved to:", log_file)

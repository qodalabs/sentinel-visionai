import sys
import os
import cv2
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# ✅ Add yolov5 folder to Python path explicitly
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
# Add both project root and yolov5 repo root to sys.path
root_str = str(ROOT_DIR)
yolo_dir = str((ROOT_DIR / "yolov5").resolve())
if root_str not in sys.path:
    sys.path.append(root_str)
if yolo_dir not in sys.path:
    sys.path.append(yolo_dir)

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.augmentations import letterbox
from yolov5.utils.general import non_max_suppression, scale_boxes
from yolov5.utils.torch_utils import select_device

# =======================
# Load Models
# =======================
device = select_device("cpu")

m1 = YOLO(str((ROOT_DIR / "models" / "best.pt").resolve()))      # YOLOv8 main weapon detector
m3 = YOLO(str((ROOT_DIR / "models" / "best3.pt").resolve()))     # YOLOv8 secondary weapon/object detector
m2 = DetectMultiBackend(str((ROOT_DIR / "models" / "best2.pt").resolve()), device=device)  # YOLOv5 general object detector
stride, names, pt = m2.stride, m2.names, m2.pt

# =======================
# Helper: IoU Calculation
# =======================
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

# =======================
# Inference Function
# =======================
def run_inference(frame):
    results_combined = []

    # ---- Model 1 (YOLOv8 weapons) ----
    results1 = m1(frame, conf=0.55)
    for box in results1[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = m1.names[cls]

        if (x2 - x1) < 30 or (y2 - y1) < 30:
            continue

        results_combined.append((x1, y1, x2, y2, conf, label, (0, 255, 0)))  # green

    # ---- Model 3 (secondary, can detect weapons and objects) ----
    results3 = m3(frame, conf=0.55)
    for box in results3[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = m3.names[cls]

        # Skip if overlapping with m1
        overlap = False
        for (xx1, yy1, xx2, yy2, _, _, _) in results_combined:
            if iou((x1, y1, x2, y2), (xx1, yy1, xx2, yy2)) > 0.3:
                overlap = True
                break

        if not overlap:
            results_combined.append((x1, y1, x2, y2, conf, label, (0, 255, 0)))  # green

    # ---- Model 2 (YOLOv5 general objects) ----
    img = letterbox(frame, 640, stride=stride)[0]
    img = img.transpose((2, 0, 1))[::-1]
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device).float()
    img /= 255
    if len(img.shape) == 3:
        img = img[None]

    pred = m2(img)
    pred = non_max_suppression(pred, 0.5, 0.45)[0]  # adjusted confidence

    if len(pred):
        pred[:, :4] = scale_boxes(img.shape[2:], pred[:, :4], frame.shape).round()
        for *xyxy, conf, cls in pred:
            x1, y1, x2, y2 = map(int, xyxy)
            label = names[int(cls)]
            results_combined.append((x1, y1, x2, y2, float(conf), label, (0, 0, 255)))  # red

    # ---- Draw results ----
    for (x1, y1, x2, y2, conf, label, color) in results_combined:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return frame

# =======================
# Live Camera Detection
# =======================
def live_detection():
    cap = cv2.VideoCapture(0)  # default webcam
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = run_inference(frame)
        cv2.imshow("Live Weapon & Object Detection", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    live_detection()

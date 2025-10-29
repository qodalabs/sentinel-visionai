import sys
import os
import cv2
import torch
from utils.augmentations import letterbox
from ultralytics import YOLO

# ---------------------------
# Add YOLOv5 repo to sys.path
# ---------------------------
YOLOV5_DIR = r"C:\weapon\yolov5"   # <--- path to the folder where you extracted yolov5
if YOLOV5_DIR not in sys.path:
    sys.path.append(YOLOV5_DIR)

from models.common import DetectMultiBackend
from utils.torch_utils import select_device
from utils.general import non_max_suppression
import numpy as np
from pathlib import Path

# Unified project root (wd-dashboard) and paths
ROOT_DIR = Path(__file__).resolve().parents[1]
YOLOV5_UNIFIED = (ROOT_DIR / "yolov5").resolve()
if str(YOLOV5_UNIFIED) not in sys.path:
    sys.path.append(str(YOLOV5_UNIFIED))

# Override model paths to unified models dir
M1_PATH = str((ROOT_DIR / "models" / "best.pt").resolve())     # Primary weapon model (YOLOv8)
M3_PATH = str((ROOT_DIR / "models" / "best3.pt").resolve())    # Secondary weapon model (YOLOv8)
M2_PATH = str((ROOT_DIR / "models" / "best2.pt").resolve())    # Non-weapon model (YOLOv5)

# ---------------------------
# Model paths
# ---------------------------
M1_PATH = r"C:\weapon\best.pt"     # Primary weapon model (YOLOv8)
M3_PATH = r"C:\weapon\best3.pt"    # Secondary weapon model (YOLOv8)
M2_PATH = r"C:\weapon\best2.pt"    # Non-weapon model (YOLOv5)

# ---------------------------
# Load models
# ---------------------------
def load_models():
    # YOLOv8 weapon models (unified paths)
    m1 = YOLO(str((ROOT_DIR / "models" / "best.pt").resolve()))
    m3 = YOLO(str((ROOT_DIR / "models" / "best3.pt").resolve()))

    # YOLOv5 general object model (unified path)
    global device
    device = select_device("cpu")
    m2 = DetectMultiBackend(str((ROOT_DIR / "models" / "best2.pt").resolve()), device=device, dnn=False, data=None, fp16=False)

    return m1, m3, m2

# ---------------------------
# Non-overlap check (IoU based)
# ---------------------------
def iou(box1, box2):
    """Compute IoU between two bounding boxes"""
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2

    xi1, yi1 = max(x1, x1g), max(y1, y1g)
    xi2, yi2 = min(x2, x2g), min(y2, y2g)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)

    union = box1_area + box2_area - inter_area
    return inter_area / union if union > 0 else 0

def is_overlapping(box, existing_boxes, threshold=0.5):
    for ex in existing_boxes:
        if iou(box, ex) > threshold:
            return True
    return False

# ---------------------------
# Run inference
# ---------------------------
def run_inference(frame, m1, m3, m2):
    # Preprocess frame once
    img = letterbox(frame, 640, stride=32, auto=True)[0]
    img = img.transpose((2, 0, 1))  # HWC -> CHW
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device).float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    results = []

    # -------- Model 1 (best.pt) --------
    pred1 = m1(img, augment=False, visualize=False)[0]
    pred1 = non_max_suppression(pred1, 0.5, 0.45)[0]
    results.extend([("best.pt", *det.cpu().numpy()) for det in pred1])

    # -------- Model 3 (best3.pt) --------
    # Skip detections overlapping with model 1
    if pred1 is not None and len(pred1):
        pred3 = m3(img, augment=False, visualize=False)[0]
        pred3 = non_max_suppression(pred3, 0.5, 0.45)[0]

        for det in pred3:
            iou_max = 0
            for det1 in pred1:
                iou = box_iou(det[:4].unsqueeze(0), det1[:4].unsqueeze(0)).item()
                iou_max = max(iou_max, iou)
            if iou_max < 0.5:  # keep only non-overlapping
                results.append(("best3.pt", *det.cpu().numpy()))

    # -------- Model 2 (best2.pt, all objects except weapons) --------
    pred2 = m2(img, augment=False, visualize=False)[0]
    pred2 = non_max_suppression(pred2, 0.25, 0.45)[0]  # low conf threshold, no filtering
    results.extend([("best2.pt", *det.cpu().numpy()) for det in pred2])

    return results


# ---------------------------
# Draw results
# ---------------------------
def draw_results(frame, results):
    for label, box in results:
        x1, y1, x2, y2 = map(int, box)
        color = (0, 0, 255) if label == "weapon" else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, color, 2)
    return frame

# ---------------------------
# Main
# ---------------------------
def main():
    m1, m3, m2 = load_models()

    cap = cv2.VideoCapture(0)  # webcam, or replace with video path
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = run_inference(frame, m1, m3, m2)
        frame = draw_results(frame, results)

        cv2.imshow("Weapon & Object Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


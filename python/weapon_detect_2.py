import cv2
import os
from pathlib import Path
from ultralytics import YOLO
from tkinter import Tk, filedialog
import torch

# Resolve project paths
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

# === Step 1: Load Both Models (unified models dir) ===
model1 = YOLO(str((ROOT_DIR / "models" / "best.pt").resolve()))
model2 = YOLO(str((ROOT_DIR / "models" / "best3.pt").resolve()))

# === Step 2: Select File ===
Tk().withdraw()
file_path = filedialog.askopenfilename(title="Select Image or Video File")
if not file_path:
    print("❌ No file selected.")
    exit()

file_ext = os.path.splitext(file_path)[1].lower()
CONF_THRESHOLD = 0.7  # Ignore model2 detections below this
IOU_THRESHOLD = 0.7   # Consider a match if IoU > this

# === IoU Calculation Helper ===
def compute_iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    iou = interArea / float(box1Area + box2Area - interArea)
    return iou

# === Draw Box Helper ===
def draw_box(img, box, label, color):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# === Process Image Frame ===
def process_frame(image):
    res1 = model1(image)
    res2 = model2(image)

    boxes1 = res1[0].boxes
    boxes2 = res2[0].boxes

    img = image.copy()

    model1_boxes = []

    # ✅ Filter + Draw high-confidence model1 detections
    for box in boxes1:
        conf = float(box.conf)
        if conf < 0.4:
            continue  # Ignore low confidence
        cls_id = int(box.cls)
        xyxy = box.xyxy[0].tolist()
        model1_boxes.append(xyxy)
        label = f"{res1[0].names[cls_id]} {conf:.2f}"
        draw_box(img, xyxy, f"M1: {label}", (255, 0, 0))

    # ✅ Filter + draw model2 only if not overlapping and confidence >= 0.4
    for box in boxes2:
        conf = float(box.conf)
        if conf < 0.4:
            continue
        cls_id = int(box.cls)
        xyxy2 = box.xyxy[0].tolist()

        overlap_found = False
        for xyxy1 in model1_boxes:
            if compute_iou(xyxy1, xyxy2) > IOU_THRESHOLD:
                overlap_found = True
                break

        if not overlap_found:
            label = f"{res2[0].names[cls_id]} {conf:.2f}"
            draw_box(img, xyxy2, f"M2: {label}", (0, 255, 0))

    return img


# === Image Mode ===
if file_ext in ['.jpg', '.jpeg', '.png']:
    img = cv2.imread(file_path)
    out_img = process_frame(img)
    out_dir = (ROOT_DIR / "output").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = str(out_dir / "combined_output.jpg")
    cv2.imwrite(out_path, out_img)
    print(f"✅ Saved output image: {out_path}")
    os.startfile(out_path)

# === Video Mode ===
elif file_ext in ['.mp4', '.avi', '.mov', '.mkv']:
    cap = cv2.VideoCapture(file_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out_dir = (ROOT_DIR / "output").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = str(out_dir / "combined_output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        processed = process_frame(frame)
        out.write(processed)

    cap.release()
    out.release()
    print(f"✅ Saved output video: {out_path}")
    os.startfile(out_path)

else:
    print("❌ Unsupported file format.")

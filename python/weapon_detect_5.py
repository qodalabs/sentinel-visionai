import sys
import os
import cv2
import torch
import numpy as np
import csv
import time
import platform
import requests
import json
from ultralytics import YOLO
from pathlib import Path

# Ensure the wd-dashboard root is on sys.path so `import yolov5` works
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
# Command-line Arguments
# =======================
MODE = None
VIDEO_PATH = None
CAMERA_ID = None
JWT_TOKEN = None

args = sys.argv[1:]
if '--webcam' in args:
    MODE = 'webcam'
elif '--live' in args:
    MODE = 'live'
    VIDEO_PATH = args[args.index('--live') + 1]  # RTSP URL

# Optional: --token <JWT>
if '--token' in args:
    JWT_TOKEN = args[args.index('--token') + 1]
if '--camera-id' in args:
    CAMERA_ID = args[args.index('--camera-id') + 1]

if not MODE:
    print("Usage: python weapon_detect.py --webcam|--live <url> [--token <JWT>]")
    sys.exit(1)

if not JWT_TOKEN:
    print("âŒ No JWT token provided. Use --token <JWT>")
    sys.exit(1)

HEADERS = {"Authorization": f"Bearer {JWT_TOKEN}"}
BACKEND_URL = "http://localhost:5000"

# Weapon labels to alert on (case-insensitive)
WEAPON_LABELS = {'weapon','gun','pistol','revolver','rifle','shotgun','knife','dagger','sword','grenade'}

# =======================
# Load Models
# =======================
device = select_device("cpu")
# Override model weights to unified wd-dashboard/models
m1 = YOLO(str((ROOT_DIR / "models" / "best.pt").resolve()))
m3 = YOLO(str((ROOT_DIR / "models" / "best3.pt").resolve()))
m2 = DetectMultiBackend(str((ROOT_DIR / "models" / "best2.pt").resolve()), device=device)
stride, names, pt = m2.stride, m2.names, m2.pt

# Unified output path
OUTPUT_DIR = str((ROOT_DIR / "output").resolve())
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALERT_LOG = os.path.join(OUTPUT_DIR, "weapon_alerts.csv")
with open(ALERT_LOG, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Time", "Frame", "Label", "Confidence"])

frame_counter = 0
skip_frames = 5  # process every 5th frame for responsiveness
# ======================= 
# Helper Functions 
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

def trigger_alert(frame_num, label, conf):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    # CSV log
    with open(ALERT_LOG, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, frame_num, label, f"{conf:.2f}"])

    # Emit JSON for server to capture and persist (avoid duplicate inserts)
    payload = {
        "weaponType": label,
        "confidence": float(conf),
        "cameraId": CAMERA_ID or ("live_camera" if MODE == 'live' else 'webcam'),
        "imageUrl": None,
        "timestamp": timestamp
    }
    print(json.dumps(payload), flush=True)
    # Optional beep
    if platform.system() == "Windows":
        import winsound
        winsound.Beep(1000, 500)
    else:
        os.system("echo -e '\a'")

def run_inference(frame):
    results_combined = []

    # =======================
    # Model 1 (best.pt)
    # =======================
    results1 = m1(frame, conf=0.4)
    for box in results1[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf_score = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = m1.names[cls_id] if hasattr(m1, "names") else "weapon"
        if label.lower() not in WEAPON_LABELS:
            continue

        # Keep all weapon labels from m1 (grenade filtered earlier)

        if (x2 - x1) < 20 or (y2 - y1) < 20:
            continue
        results_combined.append((x1, y1, x2, y2, conf_score, label, (0,0,255)))
        trigger_alert(frame_counter, label, conf_score)

    # =======================
    # Model 3 (best3.pt)
    # =======================
    results3 = m3(frame, conf=0.4)
    for box in results3[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf_score = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = m3.names[cls_id] if hasattr(m3, "names") else "weapon"

        if label.lower() != "gun":  # only "gun"
            continue

        overlap = False
        for (xx1, yy1, xx2, yy2, _, _, _) in results_combined:
            if iou((x1,y1,x2,y2),(xx1,yy1,xx2,yy2)) > 0.4:
                overlap=True
                break
        if not overlap:
            results_combined.append((x1, y1, x2, y2, conf_score, label, (0,0,255)))
            trigger_alert(frame_counter, label, conf_score)

    # =======================
    # Model 2 (best2.pt)
    # =======================
    img = letterbox(frame, 640, stride=stride)[0]
    img = img.transpose((2,0,1))[::-1]
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device).float()/255
    if len(img.shape)==3: 
        img=img[None]
    pred = m2(img)
    pred = non_max_suppression(pred, 0.45, 0.45)[0]
    if len(pred):
        pred[:, :4] = scale_boxes(img.shape[2:], pred[:, :4], frame.shape).round()
        for *xyxy, conf_score, cls in pred:
            x1, y1, x2, y2 = map(int, xyxy)
            cls_id = int(cls)
            label = names[cls_id] if cls_id < len(names) else "object"
            if label.lower() in WEAPON_LABELS:
                trigger_alert(frame_counter, label, float(conf_score))

                continue

            results_combined.append((x1, y1, x2, y2, float(conf_score), label, (0,255,0)))

    # =======================
    # Draw Final Results
    # =======================
    for (x1, y1, x2, y2, conf_score, label, color) in results_combined:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} {conf_score:.2f}", (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return frame


# =======================
# Run Live Detection
# =======================
if MODE=='webcam':
    cap = cv2.VideoCapture(0)
elif MODE=='live':
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        sys.stderr.write(f"Failed to open RTSP stream: {VIDEO_PATH}\\n")
        sys.stderr.flush()
        sys.exit(1)
else:
    print("Invalid mode")
    sys.exit(1)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            sys.stderr.write("Failed to grab frame; retrying...\\n")
            sys.stderr.flush()
            time.sleep(0.5)
            continue

        frame_counter += 1

        # âœ… Resize frame for performance
        frame = cv2.resize(frame, (800, 600))

        # âœ… Skip frames for speed
        if frame_counter % skip_frames != 0:
            continue

        processed = run_inference(frame)
        cv2.imshow("Weapon Detection", processed)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    print("Live feed terminated by server")
finally:
    cap.release()
    cv2.destroyAllWindows()








import os
from pathlib import Path
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from ultralytics import YOLO

# Resolve project paths
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent if BASE_DIR.name == 'python' else BASE_DIR

# ✅ Load YOLOv8 model
# Override model path to unified wd-dashboard/models
model = YOLO(str((ROOT_DIR / "models" / "best.pt").resolve()))

# ✅ File types
image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
video_extensions = (".mp4", ".avi", ".mov", ".mkv", ".webm")

# ✅ Select file
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(
    title="Select Image or Video",
    filetypes=[("Media files", "*.jpg *.jpeg *.png *.bmp *.webp *.mp4 *.avi *.mov *.mkv *.webm")]
)

if not file_path:
    messagebox.showinfo("No File", "No file selected.")
    exit()

ext = os.path.splitext(file_path)[1].lower()

# ✅ Image detection
if ext in image_extensions:
    image = cv2.imread(file_path)
    if image is None:
        messagebox.showerror("Error", "Could not read image.")
        exit()

    results = model.predict(source=image, conf=0.3, verbose=False)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        label = results.names[cls]
        conf = float(box.conf[0])
        color = (0, 0, 255)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, f"{label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("🟢 Weapon Detection (Image)", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ✅ Video detection
elif ext in video_extensions:
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        messagebox.showerror("Error", "Could not open video.")
        exit()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_dir = (ROOT_DIR / "output").resolve()
    os.makedirs(output_dir, exist_ok=True)
    out_path = str(output_dir / "output_video.mp4")
    out = cv2.VideoWriter(out_path, fourcc, int(cap.get(5)),
                          (int(cap.get(3)), int(cap.get(4))))

    print("🔄 Processing video...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, conf=0.3, verbose=False)[0]
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            label = results.names[cls]
            conf = float(box.conf[0])
            color = (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        out.write(frame)

    cap.release()
    out.release()
    print("✅ Video saved as output_video.mp4")
    # Also report unified output path
    print("Unified output path:", out_path)
    if os.name == "nt":
        os.system(f"start \"\" \"{out_path}\"")
    else:
        os.system(f"xdg-open '{out_path}'")

else:
    messagebox.showerror("Unsupported Format", "Please select an image or video file.")

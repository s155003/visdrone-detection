from ultralytics import YOLO
import os

model = YOLO("runs/detect/visdrone_yolov5s_run1/weights/best.pt")

# Detect whether a display is available
have_display = os.environ.get("DISPLAY") is not None

if have_display:
    print("Display found — opening live window. Press Q in the window to quit.")
    model.predict(source=0, show=True, classes=[3, 4, 5, 8, 9])
else:
    print("No display — recording 200 frames to a file instead.")
    model.predict(
        source=0,
        save=True,
        name="webcam_run",
        classes=[3, 4, 5, 8, 9],
        stream=False,
        vid_stride=1,
        max_det=300,
    )
    print("Saved to runs/detect/webcam_run/")
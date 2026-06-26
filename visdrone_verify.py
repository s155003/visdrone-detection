from ultralytics import YOLO

# Load your pretrained YOLOv5-Small
model = YOLO("yolov5su.pt")

# Short 3-epoch run: downloads VisDrone, converts labels, trains briefly.
# This is just to confirm the pipeline works end-to-end, NOT real training.
results = model.train(
    data="VisDrone.yaml",
    epochs=3,
    imgsz=640,
    name="visdrone_verify",
)
print("\n=== Verification run complete. If you see this with no errors, the pipeline works. ===")

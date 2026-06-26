from ultralytics import YOLO

model = YOLO("yolov5su.pt")

model.train(
    data="VisDrone.yaml",
    epochs=100,
    patience=15,
    imgsz=640,
    name="visdrone_yolov5s_run1",
)

print("Fine-tuning complete.")
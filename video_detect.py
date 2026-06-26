from ultralytics import YOLO

# Your fine-tuned model
model = YOLO("runs/detect/visdrone_yolov5s_run1/weights/best.pt")

# One VID sequence folder (folder of numbered .jpg frames)
sequence_folder = "datasets/VisDrone2019-VID-val/sequences/uav0000086_00000_v"

results = model.predict(
    source=sequence_folder,
    save=True,                 # saves annotated frames + stitched video
    name="vid_detect_seq1",
    classes=[3, 4, 5, 8, 9],   # vehicles only; delete this line for all 10 classes
)
print("Done. Check runs/detect/vid_detect_seq1/ for the annotated video.")
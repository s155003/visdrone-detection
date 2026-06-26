from ultralytics import YOLO
import cv2
import glob
import os

model = YOLO("runs/detect/visdrone_yolov5s_run1/weights/best.pt")

# All 7 sequence folders
seq_root = "datasets/VisDrone2019-VID-val/sequences"
sequences = sorted(os.listdir(seq_root))
print(f"Found {len(sequences)} sequences: {sequences}")

for seq in sequences:
    seq_path = os.path.join(seq_root, seq)
    run_name = f"vid_{seq}"

    # 1. Run detection on this sequence's frames
    model.predict(
        source=seq_path,
        save=True,
        name=run_name,
        classes=[3, 4, 5, 8, 9],   # vehicles only; delete for all classes
    )

    # 2. Stitch the annotated frames into an mp4
    frame_dir = f"runs/detect/{run_name}"
    frames = sorted(glob.glob(os.path.join(frame_dir, "*.jpg")))
    if not frames:
        print(f"  No frames for {seq}, skipping video.")
        continue
    h, w = cv2.imread(frames[0]).shape[:2]
    out = cv2.VideoWriter(f"{seq}_detected.mp4",
                          cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
    for f in frames:
        out.write(cv2.imread(f))
    out.release()
    print(f"  Saved {seq}_detected.mp4 ({len(frames)} frames)")

print("All done.")
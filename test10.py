from ultralytics import YOLO
import glob
import random
import os

# Load YOUR fine-tuned model
model = YOLO("runs/detect/visdrone_yolov5s_run1/weights/best.pt")

# Get all test images
all_images = glob.glob("datasets/VisDrone/images/test/*.jpg")

# Keep only images whose 4th group (trailing number) is between 145 and 170
matched = []
for path in all_images:
    name = os.path.basename(path)          # e.g. "0000006_01275_d_0000004.jpg"
    stem = name.replace(".jpg", "")        # "0000006_01275_d_0000004"
    parts = stem.split("_")                # ["0000006","01275","d","0000004"]
    if len(parts) == 4:                    # safety: only well-formed names
        fourth = int(parts[3])             # 4
        if 145 <= fourth <= 170:
            matched.append(path)

print(f"{len(matched)} images matched the 145-170 range")

# Randomly pick up to 10 of the matches
images = random.sample(matched, min(10, len(matched)))
print(f"Running on {len(images)} images")

# Run detection, vehicles only
results = model.predict(
    source=images,
    save=True,
    name="test10_range",
    classes=[3, 4, 5, 8, 9],
)
print("Done. Annotated images saved in runs/detect/test10_range/")
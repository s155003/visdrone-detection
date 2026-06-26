# VisDrone Object Detection

Object detection on aerial drone imagery using the VisDrone dataset. This project fine-tunes a YOLOv5-Small detector to identify vehicles and people in drone-captured scenes, and runs it across three input modes: static images, video sequences, and a live webcam feed. It is part of undergraduate research on object detection in unmanned aerial vehicle (UAV) imagery.

The detector is trained on VisDrone's 10 object categories (pedestrians, cars, trucks, buses, and more) and produces annotated outputs for each input mode. Trained model weights and the dataset itself are not included in this repository due to size, but are documented below and available on request.

## Capabilities

The project supports three detection modes:

(1) Static image detection. Run the detector on individual VisDrone images and save the results with bounding boxes drawn on detected objects.

<img width="960" height="540" alt="image" src="https://github.com/user-attachments/assets/88248845-7967-4563-83b2-0389ef235145" />

(2) Video detection. Run the detector frame by frame across VisDrone-VID sequences, then stitch the annotated frames into playable videos.

https://drive.google.com/file/d/1R59X1NZ5zQr8MV5xMt-ibV2fHHB1H93j/view?usp=sharing

(3) Live webcam detection. Run the detector in real time on a connected USB webcam, drawing boxes on the live feed.

<img width="302.4" height="403.2" alt="unnamed" src="https://github.com/user-attachments/assets/1da2b3d6-9d92-4ddb-ae08-0155b3f7bd70" />

## Scripts

Each script performs a single step of the pipeline.

`visdrone_train.py` — Fine-tunes YOLOv5-Small on the VisDrone training set so it learns to detect drone-view objects, then saves the trained weights.

`test10.py` — Runs the trained detector on a sample of test images and saves annotated copies. Supports filtering to vehicle classes only.

`video_detect.py` — Runs detection on a single VisDrone-VID video sequence.

`video_all.py` — Runs detection across all video sequences and stitches each into an annotated `.mp4`.

`make_video.py` — Combines a folder of annotated frames into a single playable video using OpenCV.

`webcam_detect.py` — Runs live detection from a connected USB webcam.

## Results

Training metrics for the fine-tuned YOLOv5-Small model are in the `results/` folder.

`results.png` — Loss and mAP curves over training.

`confusion_matrix.png` — Per-class confusion between categories.

`BoxPR_curve.png` — Precision–recall curve.

`results.csv` — Raw per-epoch metric values.

## Classes

VisDrone defines 10 object categories:

`0` pedestrian, `1` people, `2` bicycle, `3` car, `4` van, `5` truck, `6` tricycle, `7` awning-tricycle, `8` bus, `9` motor.

Vehicle-only detection (used in several scripts) filters to classes `3, 4, 5, 8, 9`.

## Setup

Requires Python 3.10+ and the Ultralytics library.

```bash
pip install ultralytics
```

The VisDrone dataset is not included in this repository (several GB). The static-image dataset (Task 1) downloads automatically via Ultralytics' built-in `VisDrone.yaml` on the first training run. The video dataset (Task 2, VisDrone-VID) is downloaded manually from the official VisDrone repository.

## Usage

Fine-tune on VisDrone static images:

```bash
python visdrone_train.py
```

Run detection on test images:

```bash
python test10.py
```

Run detection on all video sequences:

```bash
python video_all.py
```

Run live webcam detection:

```bash
python webcam_detect.py
```

## Notes

Trained model weights (`.pt`) and datasets are excluded via `.gitignore` to keep the repository lightweight; weights are available on request. The detector is trained on aerial (top-down) imagery, so detection on ground-level scenes such as a desk-height webcam will be less accurate due to the domain gap.

## Dataset

This project uses the VisDrone dataset:

> VisDrone-Dataset — https://github.com/VisDrone/VisDrone-Dataset

import cv2
import glob
import os

# Folder of annotated frames from the detection run
frame_dir = "runs/detect/vid_detect_seq1"
frames = sorted(glob.glob(os.path.join(frame_dir, "*.jpg")))
print(f"Found {len(frames)} frames")

# Read the first frame to get dimensions
first = cv2.imread(frames[0])
height, width = first.shape[:2]

# Set up the video writer (mp4, 30 fps)
out = cv2.VideoWriter(
    "seq1_detected.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    30,
    (width, height),
)

# Write every frame into the video
for f in frames:
    out.write(cv2.imread(f))

out.release()
print("Done. Video saved as seq1_detected.mp4")
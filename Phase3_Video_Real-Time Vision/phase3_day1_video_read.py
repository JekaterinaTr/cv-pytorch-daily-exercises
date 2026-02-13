"""
PHASE 3 — Video & Real-Time Vision
Day 1: Read Video Files with OpenCV

Concepts:
- Videos are sequences of frames (images)
- cv2.VideoCapture opens video files
- Accessing frames with .read()
- Video properties: frame count, width, height, FPS
"""

import cv2
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load video
# -----------------------------
video_path = "sample_video.mp4"  # Replace with your video path
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise FileNotFoundError(f"Cannot open video file: {video_path}")

print("Video opened successfully!")

# -----------------------------
# 2. Read video properties
# -----------------------------
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

print(f"Total frames: {total_frames}")
print(f"Frame size: {width}x{height}")
print(f"FPS: {fps}")

# -----------------------------
# 3. Read and display first frame
# -----------------------------
ret, frame = cap.read()  # ret = True if frame read successfully

if not ret:
    raise ValueError("Cannot read first frame")

# Display using Matplotlib (convert BGR -> RGB)
plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
plt.title("First Frame")
plt.axis("off")
plt.show()

# -----------------------------
# 4. Optional: Loop through first few frames
# -----------------------------
frame_count = 0
max_display = 5  # Display only first 5 frames for demonstration

while True:
    ret, frame = cap.read()
    if not ret:
        break  # End of video
    frame_count += 1
    if frame_count <= max_display:
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.title(f"Frame {frame_count}")
        plt.axis("off")
        plt.show()

print(f"Total frames read: {frame_count}")

# -----------------------------
# 5. Release resources
# -----------------------------
cap.release()
cv2.destroyAllWindows()
print("Video capture released and windows destroyed")

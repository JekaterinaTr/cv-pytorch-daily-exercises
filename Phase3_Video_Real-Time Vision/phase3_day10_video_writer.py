"""
PHASE 3 — Video & Real-Time Vision
Day 10: VideoWriter & Full Pipeline

Concepts:
- VideoCapture
- VideoWriter
- Frame processing
- Saving processed video
- Clean shutdown
"""

import cv2

# ----------------------------------
# 1. Open Webcam
# ----------------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

print("Webcam opened successfully.")

# ----------------------------------
# 2. Get Frame Properties
# ----------------------------------

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 20  # manually set stable FPS

print(f"Resolution: {width} x {height}")
print(f"FPS: {fps}")

# ----------------------------------
# 3. Create VideoWriter
# ----------------------------------

# Codec (XVID is widely supported)
fourcc = cv2.VideoWriter_fourcc(*'XVID')

out = cv2.VideoWriter(
    "output_day10.avi",  # output file
    fourcc,
    fps,
    (width, height)
)

print("Recording started. Press 'q' to stop.")

# ----------------------------------
# 4. Main Processing Loop
# ----------------------------------

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # ----------------------------------
    # 5. Apply Frame Transformations
    # ----------------------------------

    # Example 1: Flip horizontally
    frame = cv2.flip(frame, 1)

    # Example 2: Add text overlay
    cv2.putText(
        frame,
        "Phase 3 - Day 10 Recording",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # ----------------------------------
    # 6. Save Frame
    # ----------------------------------

    out.write(frame)

    # ----------------------------------
    # 7. Display Frame
    # ----------------------------------

    cv2.imshow("Recording", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Stopping recording...")
        break

# ----------------------------------
# 8. Cleanup
# ----------------------------------

cap.release()
out.release()
cv2.destroyAllWindows()

print("Video saved as output_day10.avi")
print("Resources released successfully.")